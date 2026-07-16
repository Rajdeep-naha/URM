from transformers import LlamaForSequenceClassification, Cache
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from .mdn_head import URMMDNHead, mixture_mean

class Weights(torch.nn.Module):
    """
    Gating network matching original URM gating layers.
    """
    def __init__(self, hidden_size=4096):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.SELU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.SELU(),
            torch.nn.Linear(hidden_size, 5)
        )

    def forward(self, x):
        # Cast input to match the linear layers' weight dtype dynamically
        return self.fc(x.to(self.fc[0].weight.dtype))


class LlamaForSequenceClassificationWithMDN(LlamaForSequenceClassification):
    """
    Custom LLaMA model wrapper that implements the MDN attribute head
    and gating network. Supports both 'label' and 'residual' target modes.
    """
    def __init__(self, config, **kwargs):
        # Allow uncertainty_target to be passed via config or kwargs
        if "uncertainty_target" in kwargs:
            config.uncertainty_target = kwargs.pop("uncertainty_target")
        elif not hasattr(config, "uncertainty_target"):
            config.uncertainty_target = "label"

        super().__init__(config)
        
        # Override the sequence classification score head with URMMDNHead
        num_components = getattr(config, "num_components", 3)
        use_gaussian = getattr(config, "use_gaussian", False)
        self.score = URMMDNHead(
            hidden_size=config.hidden_size,
            num_attributes=5,
            num_components=num_components,
            use_gaussian=use_gaussian,
            uncertainty_target=config.uncertainty_target
        )
        
        # Add the gating network Weights
        self.weights = Weights(hidden_size=config.hidden_size)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = transformer_outputs[0]  # shape: [B, L, hidden_size]
        
        # Extract sequence length (pooling index)
        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(hidden_states.device)
            else:
                sequence_lengths = -1

        # 1. Forward pass of the Gating network
        weights = self.weights(hidden_states.to(torch.float16))  # shape: [B, L, 5]
        if isinstance(sequence_lengths, torch.Tensor):
            pooled_weights = weights[torch.arange(batch_size, device=weights.device), sequence_lengths]
        else:
            pooled_weights = weights[:, sequence_lengths]

        # 2. Forward pass of the MDN Head / Gaussian Head
        uncertainty_target = getattr(self.score, "uncertainty_target", "label")
        use_gaussian = getattr(self.score, "use_gaussian", False)

        if use_gaussian:
            if uncertainty_target == "residual":
                mean_out, mu, sigma = self.score(hidden_states.to(self.score.proj.weight.dtype))
                if isinstance(sequence_lengths, torch.Tensor):
                    pooled_mean = mean_out[torch.arange(batch_size, device=mean_out.device), sequence_lengths]
                    pooled_mu = mu[torch.arange(batch_size, device=mu.device), sequence_lengths]
                    pooled_sigma = sigma[torch.arange(batch_size, device=sigma.device), sequence_lengths]
                else:
                    pooled_mean = mean_out[:, sequence_lengths]
                    pooled_mu = mu[:, sequence_lengths]
                    pooled_sigma = sigma[:, sequence_lengths]
                
                expected_attribute_rewards = pooled_mean
                score_params = (pooled_mu, pooled_sigma)
            else:
                mu, sigma = self.score(hidden_states.to(self.score.proj.weight.dtype))
                if isinstance(sequence_lengths, torch.Tensor):
                    pooled_mu = mu[torch.arange(batch_size, device=mu.device), sequence_lengths]
                    pooled_sigma = sigma[torch.arange(batch_size, device=sigma.device), sequence_lengths]
                else:
                    pooled_mu = mu[:, sequence_lengths]
                    pooled_sigma = sigma[:, sequence_lengths]
                
                expected_attribute_rewards = pooled_mu
                score_params = (pooled_mu, pooled_sigma)
        else:
            if uncertainty_target == "residual":
                mean_out, pi, mu, s = self.score(hidden_states.to(self.score.proj.weight.dtype))
                if isinstance(sequence_lengths, torch.Tensor):
                    pooled_mean = mean_out[torch.arange(batch_size, device=mean_out.device), sequence_lengths]
                    pooled_pi = pi[torch.arange(batch_size, device=pi.device), sequence_lengths]
                    pooled_mu = mu[torch.arange(batch_size, device=mu.device), sequence_lengths]
                    pooled_s = s[torch.arange(batch_size, device=s.device), sequence_lengths]
                else:
                    pooled_mean = mean_out[:, sequence_lengths]
                    pooled_pi = pi[:, sequence_lengths]
                    pooled_mu = mu[:, sequence_lengths]
                    pooled_s = s[:, sequence_lengths]
                
                expected_attribute_rewards = pooled_mean
                score_params = (pooled_pi, pooled_mu, pooled_s)
            else:
                pi, mu, s = self.score(hidden_states.to(self.score.proj.weight.dtype))
                if isinstance(sequence_lengths, torch.Tensor):
                    pooled_pi = pi[torch.arange(batch_size, device=pi.device), sequence_lengths]
                    pooled_mu = mu[torch.arange(batch_size, device=mu.device), sequence_lengths]
                    pooled_s = s[torch.arange(batch_size, device=s.device), sequence_lengths]
                else:
                    pooled_pi = pi[:, sequence_lengths]
                    pooled_mu = mu[:, sequence_lengths]
                    pooled_s = s[:, sequence_lengths]
                
                expected_attribute_rewards = mixture_mean(pooled_pi, pooled_mu)
                score_params = (pooled_pi, pooled_mu, pooled_s)

        # 3. Final Reward Score
        scores = (expected_attribute_rewards * pooled_weights.to(expected_attribute_rewards.dtype)).sum(dim=-1).view(-1, 1)

        loss = None
        if not return_dict:
            return scores, pooled_weights, expected_attribute_rewards, score_params

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=scores,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
