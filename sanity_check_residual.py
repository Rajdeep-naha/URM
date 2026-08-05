import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from datasets import load_dataset
from models.modeling_mdn_urm import LlamaForSequenceClassificationWithMDN
import argparse

def main():
    device = "cuda"
    dtype = torch.bfloat16
    
    print("Loading model...")
    model = LlamaForSequenceClassificationWithMDN.from_pretrained(
        "/localstorage/home/f20221218/URM-LLaMa-3.1-8B",
        ignore_mismatched_sizes=True,
        torch_dtype=dtype,
        device_map="auto",
        uncertainty_target="residual"
    )
    
    print("Loading residual MDN weights...")
    model.score.load_state_dict(torch.load("checkpoints/stage1/best_mdn_head_residual.pt", map_location="cpu"))
    
    tokenizer = AutoTokenizer.from_pretrained("/localstorage/home/f20221218/URM-LLaMa-3.1-8B")
    dataset = load_dataset("allenai/reward-bench", split="filtered")
    
    # We will FORCE the gating network to output uniform weights
    # To do this, we can just patch the forward method of the weights module
    def uniform_forward(hidden_states):
        batch, seq, _ = hidden_states.shape
        return torch.ones(batch, seq, 5, device=hidden_states.device, dtype=hidden_states.dtype) / 5.0
        
    model.weights.forward = uniform_forward
    model.eval()
    
    correct = 0
    total = min(100, len(dataset))
    
    print(f"Evaluating {total} samples with UNIFORM gating weights...")
    with torch.no_grad():
        for i in range(total):
            item = dataset[i]
            
            chosen_conv = [{"role": "user", "content": item["prompt"]}, {"role": "assistant", "content": item["chosen"]}]
            rej_conv = [{"role": "user", "content": item["prompt"]}, {"role": "assistant", "content": item["rejected"]}]
            
            chosen_text = tokenizer.apply_chat_template(chosen_conv, tokenize=False)
            rej_text = tokenizer.apply_chat_template(rej_conv, tokenize=False)
            
            c_inp = tokenizer(chosen_text, return_tensors="pt").to(device)
            r_inp = tokenizer(rej_text, return_tensors="pt").to(device)
            
            c_score = model(**c_inp, return_dict=False)[0].item()
            r_score = model(**r_inp, return_dict=False)[0].item()
            
            if c_score > r_score:
                correct += 1
                
    print(f"Accuracy with UNIFORM gating weights: {correct/total*100:.2f}%")

if __name__ == "__main__":
    main()
