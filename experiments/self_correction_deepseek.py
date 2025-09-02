"""
Code for using DeepSeek-R1-distill-qwen-1.5b to output reasoning towards
correct answer later used for SFT.
"""

import os
from datasets import load_dataset
from args import get_args_self_correction_deepseek
from dataset import format_dataset_entry_self_correction
from inference import batch_inference_llm
from llm import load_llm
import transformers
import torch
import json 

transformers.set_seed(42)

if __name__=="__main__":
    args = get_args_self_correction_deepseek()

    # Model loading with optimizations to speed up inference
    print("Loading and setting LLM to evaluation mode.")
    original_model, tokenizer = load_llm(model_path=args.model_path, quantized=args.quantized)
    original_model.eval()
    print("Compiling the model for faster inference... (this may take a minute)")
    compiled_model = torch.compile(original_model, mode="reduce-overhead")
    print("Model compiled successfully.")

    # Loading and applying chat template on test set
    print("Loading test dataset...")
    dataset = load_dataset('json', data_files=args.data_file_path, split='train')
    print(f"Original dataset size: {len(dataset)}")

    self_correct_dataset = dataset.filter(lambda example: not example['correct_or_not'])
    print(f"Filtered dataset size (incorrect answers only): {len(self_correct_dataset)}")

    self_correct_formatted_dataset = self_correct_dataset.map(format_dataset_entry_self_correction)

    print("\n--- Example of Correctly Formatted Text for Self-Correction ---")
    print(self_correct_formatted_dataset[0]['text'])
    print("-------------------------------------------------------\n")

    # Resumability: Load IDs that have already been processed 
    processed_ids = set()
    if os.path.exists(os.path.join('/kaggle/working', args.output_file)):
        with open(args.output_file, 'r') as f:
            for line in f:
                try:
                    processed_ids.add(json.loads(line)['id'])
                except (json.JSONDecodeError, KeyError):
                    # Handle corrupted lines or lines without an 'id'
                    print(f"Skipping corrupted line: {line.strip()}")

    print(f"Found {len(processed_ids)} already processed IDs in {args.output_file}.")
    unprocessed_dataset = self_correct_formatted_dataset.filter(
        lambda example: example['id'] not in processed_ids
    )

    batch_inference_llm(model=compiled_model, tokenizer=tokenizer, dataset=unprocessed_dataset, output_file_path=args.output_file)

