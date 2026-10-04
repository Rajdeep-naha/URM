from transformers import BitsAndBytesConfig
import torch
from datasets import load_dataset
from rewarduq.methods.residual_mdn.residual_mdn_model import ResidualMDNModelConfig, ResidualMDNModel
from rewarduq.methods.residual_mdn.residual_mdn_pipeline import ResidualMDNPipeline
from rewarduq.methods.residual_mdn.residual_mdn_trainer import ResidualMDNTrainerConfig
import os

def main():
    print("Loading dataset...")
    # Load our joined dataset
    dataset = load_dataset("json", data_files="../ALD/../ALD/rewarduq/data/joined_pairs.jsonl")
    
    # Split into train/val
    split = dataset["train"].train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    
    print(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")
    
    # Configure model for QLoRA
    model_config = ResidualMDNModelConfig(
        base_model_name_or_path="/localstorage/home/f20221218/URM-LLaMa-3.1-8B",
        base_model_class="AutoModelForCausalLM",
        base_model_init_kwargs={
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        }, # wait, what should we use for causal LM?
        # QLoRA configuration
        peft_config={
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "v_proj"]
        },
        num_attributes=5,
        num_components=3
    )
    
    # Trainer config
    trainer_config = ResidualMDNTrainerConfig(
        output_dir="/localstorage/home/f20221218/URM/results/qlora_residual_mdn",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        num_train_epochs=1,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        lambda_kl=1.0,
        bf16=True, # Llama 3 supports bf16
        remove_unused_columns=False,
    )
    
    print("Initializing pipeline...")
    pipeline = ResidualMDNPipeline(
        model_config=model_config,
        trainer_config=trainer_config
    )
    
    print("Starting training...")
    pipeline.train(train_dataset, eval_dataset)
    
if __name__ == "__main__":
    main()
