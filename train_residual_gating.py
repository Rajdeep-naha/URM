import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, LlamaConfig, get_cosine_schedule_with_warmup
from datasets import load_dataset
import wandb
import numpy as np
from tqdm import tqdm

from models.modeling_mdn_urm import LlamaForSequenceClassificationWithMDN


class PreferenceDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        if "prompt" in item:
            prompt = item["prompt"]
            chosen = item["chosen"]
            rejected = item["rejected"]

            # Format using chat template for chosen
            chosen_conv = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": chosen}
            ]
            # Format using chat template for rejected
            rejected_conv = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": rejected}
            ]
        else:
            # Skywork dataset structure (lists of message dicts)
            chosen_conv = item["chosen"]
            rejected_conv = item["rejected"]

        chosen_text = self.tokenizer.apply_chat_template(chosen_conv, tokenize=False)
        rejected_text = self.tokenizer.apply_chat_template(rejected_conv, tokenize=False)

        chosen_inputs = self.tokenizer(
            chosen_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        rejected_inputs = self.tokenizer(
            rejected_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "chosen_input_ids": chosen_inputs["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_inputs["attention_mask"].squeeze(0),
            "rejected_input_ids": rejected_inputs["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_inputs["rejected_attention_mask" if "rejected_attention_mask" in rejected_inputs else "attention_mask"].squeeze(0),
        }



def parse_args():
    parser = argparse.ArgumentParser(description="Stage 2: Gating Network Learning")
    parser.add_argument("--model_name_or_path", type=str, default="LxzGordon/URM-LLaMa-3.1-8B", help="Model checkpoint path or HF model id")
    parser.add_argument("--mdn_head_weights", type=str, default="checkpoints/stage1/best_mdn_head.pt", help="Path to trained MDN head state dict")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per device (usually 1 due to high memory consumption of pairs)")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--grad_accum_steps", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--save_dir", type=str, default="checkpoints/residual_gating", help="Checkpoint directory")
    parser.add_argument("--dry_run", action="store_true", help="Run a quick CPU dry-run check")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--device_map", type=str, default="auto", help="Device map configuration")
    parser.add_argument("--num_components", type=int, default=3, help="Number of mixture components")
    parser.add_argument("--gaussian", action="store_true", help="Use a single Gaussian head baseline")
    return parser.parse_args()


def remove_hooks_and_materialize_meta_parameters(model, device):
    from accelerate.hooks import remove_hook_from_module
    print("Resolving meta parameters and removing offload hooks...")
    
    for module_name in ["score", "weights"]:
        if not hasattr(model, module_name):
            continue
        module = getattr(model, module_name)
        
        # Remove any accelerate hooks recursively
        remove_hook_from_module(module, recurse=True)
        
        # Materialize meta parameters recursively
        def materialize_submodule(submod):
            for param_name, param in list(submod.named_parameters(recurse=False)):
                if param.device.type == "meta":
                    new_param = nn.Parameter(torch.empty_like(param, device=device))
                    # Initialize
                    if "weight" in param_name:
                        nn.init.xavier_uniform_(new_param)
                    else:
                        nn.init.zeros_(new_param)
                    submod.register_parameter(param_name, new_param)
                    print(f"Materialized meta parameter: {param_name} in {submod.__class__.__name__} to {device}")
            
            for child in submod.children():
                materialize_submodule(child)
                
        materialize_submodule(module)
        module.to(device)


def main():
    args = parse_args()

    if args.wandb:
        wandb.init(project="mdn-urm-stage2", name="gating-network")

    os.makedirs(args.save_dir, exist_ok=True)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Initialize tokenizer
    if args.dry_run:
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-LlamaForCausalLM")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # 2. Initialize Model
    if args.dry_run:
        print("Initializing tiny config LLaMA model for dry run...")
        config = LlamaConfig(
            vocab_size=len(tokenizer),
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            pad_token_id=tokenizer.pad_token_id or 0,
            bos_token_id=tokenizer.bos_token_id or 1,
            eos_token_id=tokenizer.eos_token_id or 2,
            num_components=args.num_components,
            use_gaussian=args.gaussian,
            uncertainty_target="residual"
        )
        model = LlamaForSequenceClassificationWithMDN(config)
    else:
        print(f"Loading model with MDN Head from {args.model_name_or_path}...")
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_skip_modules=["weights", "score"]
        )
        
        model = LlamaForSequenceClassificationWithMDN.from_pretrained(
            args.model_name_or_path,
            ignore_mismatched_sizes=True,
            torch_dtype=dtype,
            device_map="auto",
            quantization_config=quantization_config,
            num_components=args.num_components,
            use_gaussian=args.gaussian,
            uncertainty_target="residual",
            load_in_4bit=True
        )
        
        # Check device of score.proj.weight or weights.fc
        if model.score.proj.weight.device.type == "meta" or model.weights.fc[0].weight.device.type == "meta":
            print("[WARNING] Custom parameters are on meta device! Materializing to target device...")
            remove_hooks_and_materialize_meta_parameters(model, device)
            
        # Load trained MDN Head weights if file exists
        if os.path.exists(args.mdn_head_weights):
            print(f"Loading trained MDN Head weights from {args.mdn_head_weights}...")
            model.score.load_state_dict(torch.load(args.mdn_head_weights, map_location="cpu"), assign=True)
        else:
            print(f"Warning: Trained MDN Head weights not found at {args.mdn_head_weights}. Using randomly initialized head.")
            
        print("Loading original gating weights for initialization...")
        original_gating_weights = "checkpoints/stage2/best_gating_weights.pt"
        if os.path.exists(original_gating_weights):
            state_dict = torch.load(original_gating_weights, map_location="cpu")
            model.weights.load_state_dict(state_dict, assign=True)
            print("Loaded gating weights successfully.")
        else:
            print(f"Warning: {original_gating_weights} not found. Cannot initialize residual gating network.")

    if not hasattr(model, "hf_device_map"):
        model.to(device)

    # 3. Freeze Backbone and MDN head, Unfreeze Gating layer (model.weights)
    print("Freezing LLaMA backbone and MDN head score...")
    for param in model.model.parameters():
        param.requires_grad = False
    for param in model.score.parameters():
        param.requires_grad = False

    print("Unfreezing Gating network (weights)...")
    for param in model.weights.parameters():
        param.requires_grad = True

    # Check trainable parameters
    trainable_params = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f"Trainable parameters: {trainable_params}")

    # 4. Load Dataset
    if args.dry_run:
        print("Generating mock preference data for dry run...")
        # Create a mock preference dataset
        mock_data = [
            {
                "prompt": "Which city is the capital of France?",
                "chosen": "Paris is the capital of France.",
                "rejected": "London is the capital of France."
            },
            {
                "prompt": "What is 2+2?",
                "chosen": "2+2 is equal to 4.",
                "rejected": "2+2 is 5."
            }
        ] * 4  # 8 samples
        train_dataset = PreferenceDataset(mock_data, tokenizer, max_length=args.max_length)
        val_dataset = PreferenceDataset(mock_data, tokenizer, max_length=args.max_length)
    else:
        print("Loading Skywork-Reward-Preference-80K-v0.1 dataset from Hugging Face...")
        # Skywork dataset is usually on "train" split
        dataset = load_dataset("Skywork/Skywork-Reward-Preference-80K-v0.1")
        # Split into train/val manually (90/10 split)
        dataset_split = dataset["train"].train_test_split(test_size=0.1, seed=42)
        train_dataset = PreferenceDataset(dataset_split["train"], tokenizer, max_length=args.max_length)
        val_dataset = PreferenceDataset(dataset_split["test"], tokenizer, max_length=args.max_length)

    # Use multiple workers and pin memory to speed up batch prep and tokenization
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )

    # 5. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.weights.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    total_steps = len(train_loader) * args.epochs // args.grad_accum_steps
    warmup_steps = int(total_steps * 0.1)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # 6. Training Loop (Bradley-Terry Loss)
    print("Starting gating training...")
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for step, batch in enumerate(progress_bar):
            # 1. Forward chosen
            chosen_input_ids = batch["chosen_input_ids"].to(device)
            chosen_attention_mask = batch["chosen_attention_mask"].to(device)
            
            chosen_outputs = model(input_ids=chosen_input_ids, attention_mask=chosen_attention_mask, return_dict=False)
            chosen_score = chosen_outputs[0]  # shape: [B, 1]

            # 2. Forward rejected
            rejected_input_ids = batch["rejected_input_ids"].to(device)
            rejected_attention_mask = batch["rejected_attention_mask"].to(device)
            
            rejected_outputs = model(input_ids=rejected_input_ids, attention_mask=rejected_attention_mask, return_dict=False)
            rejected_score = rejected_outputs[0]  # shape: [B, 1]

            # 3. Bradley-Terry Loss: -log_sigmoid(chosen_score - rejected_score)
            loss = -F.logsigmoid(chosen_score - rejected_score).mean()
            loss = loss / args.grad_accum_steps
            loss.backward()

            epoch_loss += loss.item() * args.grad_accum_steps

            if (step + 1) % args.grad_accum_steps == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.weights.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            progress_bar.set_postfix({"loss": loss.item() * args.grad_accum_steps})

            if args.wandb:
                wandb.log({
                    "train_step_loss": loss.item() * args.grad_accum_steps,
                    "lr": optimizer.param_groups[0]["lr"]
                })

        avg_train_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch + 1} average training BT loss: {avg_train_loss:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for batch in val_loader:
                chosen_input_ids = batch["chosen_input_ids"].to(device)
                chosen_attention_mask = batch["chosen_attention_mask"].to(device)
                
                chosen_outputs = model(input_ids=chosen_input_ids, attention_mask=chosen_attention_mask, return_dict=False)
                chosen_score = chosen_outputs[0]

                rejected_input_ids = batch["rejected_input_ids"].to(device)
                rejected_attention_mask = batch["rejected_attention_mask"].to(device)
                
                rejected_outputs = model(input_ids=rejected_input_ids, attention_mask=rejected_attention_mask, return_dict=False)
                rejected_score = rejected_outputs[0]

                loss = -F.logsigmoid(chosen_score - rejected_score).mean()
                val_loss += loss.item()

                # Accuracy of prioritizing chosen over rejected
                correct_predictions += (chosen_score > rejected_score).sum().item()
                total_predictions += chosen_score.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct_predictions / total_predictions
        
        print(f"Epoch {epoch + 1} validation BT loss: {avg_val_loss:.4f}")
        print(f"Epoch {epoch + 1} validation Preference Accuracy: {val_accuracy:.4f}")

        if args.wandb:
            wandb.log({
                "epoch": epoch + 1,
                "val_bt_loss": avg_val_loss,
                "val_preference_accuracy": val_accuracy
            })

        # Save checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(args.save_dir, "best_gating_weights.pt")
            print(f"Saving best gating model checkpoint to {checkpoint_path}")
            torch.save(model.weights.state_dict(), checkpoint_path)

    print("Stage 2 Training Complete!")


if __name__ == "__main__":
    main()
