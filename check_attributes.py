import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from models.modeling_mdn_urm import LlamaForSequenceClassificationWithMDN
import argparse

def remove_hooks_and_materialize_meta_parameters(model, device):
    import torch.nn as nn
    from accelerate.hooks import remove_hook_from_module
    for module_name in ["score", "weights"]:
        if not hasattr(model, module_name): continue
        module = getattr(model, module_name)
        remove_hook_from_module(module, recurse=True)
        def materialize_submodule(submod):
            for param_name, param in list(submod.named_parameters(recurse=False)):
                if param.device.type == "meta":
                    new_param = nn.Parameter(torch.empty_like(param, device=device))
                    if "weight" in param_name: nn.init.xavier_uniform_(new_param)
                    else: nn.init.zeros_(new_param)
                    submod.register_parameter(param_name, new_param)
            for child in submod.children(): materialize_submodule(child)
        materialize_submodule(module)
        module.to(device)

def main():
    device = "cuda"
    dtype = torch.bfloat16
    
    model = LlamaForSequenceClassificationWithMDN.from_pretrained(
        "/localstorage/home/f20221218/URM-LLaMa-3.1-8B",
        ignore_mismatched_sizes=True, torch_dtype=dtype, device_map="auto", uncertainty_target="residual"
    )
    remove_hooks_and_materialize_meta_parameters(model, device)
    
    model.score.load_state_dict(torch.load("checkpoints/stage1/best_mdn_head_residual.pt", map_location="cpu"))
    
    tokenizer = AutoTokenizer.from_pretrained("/localstorage/home/f20221218/URM-LLaMa-3.1-8B")
    dataset = load_dataset("allenai/reward-bench", split="filtered")
    model.eval()
    
    correct_uniform = 0
    total = min(100, len(dataset))
    
    with torch.no_grad():
        for i in range(total):
            item = dataset[i]
            chosen_conv = [{"role": "user", "content": item["prompt"]}, {"role": "assistant", "content": item["chosen"]}]
            rej_conv = [{"role": "user", "content": item["prompt"]}, {"role": "assistant", "content": item["rejected"]}]
            
            c_inp = tokenizer(tokenizer.apply_chat_template(chosen_conv, tokenize=False), return_tensors="pt").to(device)
            r_inp = tokenizer(tokenizer.apply_chat_template(rej_conv, tokenize=False), return_tensors="pt").to(device)
            
            _, _, c_attr, _ = model(**c_inp, return_dict=False)
            _, _, r_attr, _ = model(**r_inp, return_dict=False)
            
            c_score = c_attr.mean().item()
            r_score = r_attr.mean().item()
            if c_score > r_score: correct_uniform += 1
                
    print(f"Accuracy with UNIFORM gating weights: {correct_uniform/total*100:.2f}%")

if __name__ == "__main__":
    main()
