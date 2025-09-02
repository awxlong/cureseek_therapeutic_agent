#!/usr/bin/env python3
"""
Classes and helper functions for supervised fine-tuning

Code is written to run on Kaggle
"""


from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer, SFTConfig 


from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoTokenizer,
) 
import torch
import os
from CFG import *


def supervised_finetune(model, tokenizer, dataset):
    # --- PEFT & LoRA Configuration ---
    peft_config = LoraConfig(
        lora_alpha=32,
        lora_dropout=0.1,
        r=16,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # Wrap the base model with PEFT model (Attach the "Sticky Notes")
    print("Attaching LoRA adapters to the model...")
    model = get_peft_model(model, peft_config)
    print("LoRA adapters attached.")


    # --- SFT Configuration ---
    # trl's current version SFTConfig to hold ALL arguments
    # that were previously in TrainingArguments

    sft_config = SFTConfig(
        output_dir=CFG.OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        optim="paged_adamw_8bit",
        save_steps=100,
        logging_steps=10,
        learning_rate=2e-4,
        num_train_epochs=4, 
        max_grad_norm=0.3,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        fp16=True, # Kaggle T4/P100 GPUs use fp16
        gradient_checkpointing=True,
        dataset_text_field="text",
        max_seq_length=1024,
        packing=use_packing, 
    )

    # --- Initialize the Trainer and Start Training ---

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=sft_config, 
        train_dataset=dataset,
    )

    trainer.label_smoother = None



    print("Starting SFT training...")
    trainer.train()
    print("Training finished!")

    # --- Save the Final Adapter ---
    final_adapter_path = os.path.join(CFG.OUTPUT_DIR, "final_adapter")
    trainer.model.save_pretrained(final_adapter_path)
    print(f"SFT LoRA adapter saved to {final_adapter_path}")

    # --- Merge the adapter and perform inference¶

    ADAPTER_PATH = os.path.join('/kaggle/working', final_adapter_path) # Path from the previous training step
    MERGED_MODEL_PATH = "/kaggle/working/cureseek-sft-v1-merged" # Where to save the new model

    # --- Load Base Model and Tokenizer ---
    print("Loading base model...")
    # Load the base model in a higher precision for merging.
    # bfloat16 is a good choice for performance and memory.
    original_model = AutoModelForCausalLM.from_pretrained(
        CFG.MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(CFG.MODEL_NAME, 
                                            trust_remote_code=True
                                            )

    # --- Load LoRA Adapter and Merge ---
    print(f"Loading LoRA adapter from {ADAPTER_PATH}...")
    # Load the PeftModel by combining the base model with the adapter
    model = PeftModel.from_pretrained(original_model, ADAPTER_PATH)

    print("Merging adapter weights with the base model...")
    # This method merges the LoRA weights into the base model's weights
    merged_model = model.merge_and_unload()
    print("Merge complete.")

    # --- Save the Merged Model ---
    # Now you have a standalone model that can be used without the PEFT library
    print(f"Saving merged model to {MERGED_MODEL_PATH}...")
    merged_model.save_pretrained(MERGED_MODEL_PATH)
    tokenizer.save_pretrained(MERGED_MODEL_PATH)
    print("Merged model and tokenizer saved successfully!")