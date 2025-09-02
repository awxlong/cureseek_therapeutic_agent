"""
Code for running inference with quantized DeepSeek-R1 on test questions. 
"""
import json
from datasets import load_dataset
from args import get_args_deepseek_inference
from dataset import format_dataset_entry_for_inference
from inference import batch_inference_llm
from llm import load_finetuned_llm, load_llm
import os
from CFG import *

if __name__ == "__main__":

    args = get_args_deepseek_inference()

    if args.load_finetuned:
        # --- Load the Fine-Tuned Model and Tokenizer ---
        print(f"Loading fine-tuned model from {args.model_path}...")
        model, tokenizer = load_finetuned_llm(model_path=args.model_path, quantized=args.quantized)
        model.eval() 

    else:
        # Model loading with optimizations to speed up inference
        print("Loading and setting LLM to evaluation mode.")
        model, tokenizer = load_llm(model_path=args.model_path, quantized=args.quantized)
        model.eval()
        print("Compiling the model for faster inference... (this may take a minute)")
        model = torch.compile(model, mode="reduce-overhead")
        print("Model compiled successfully.")

    # Loading and applying chat template on test set
    print("Loading test dataset...")
    dataset = load_dataset('json', data_files=args.data_file_path, split='train')
    print(f"Original dataset size: {len(dataset)}")

    formatted_dataset_test = dataset.map(lambda example: format_dataset_entry_for_inference(example, tokenizer))

    print("\n--- Example of Correctly Formatted Test Set Sample Text ---")
    print(formatted_dataset_test[0]['text'])
    print("-------------------------------------------------------\n")

    
    # Resumability: Load IDs that have already been processed 
    processed_ids = set()
    if os.path.exists(args.output_file):
        with open(args.output_file, 'r') as f:
            for line in f:
                try:
                    processed_ids.add(json.loads(line)['id'])
                except (json.JSONDecodeError, KeyError):
                    # Handle corrupted lines or lines without an 'id'
                    print(f"Skipping corrupted line: {line.strip()}")
    print(f"Found {len(processed_ids)} already processed IDs in {args.output_file}.")
    unprocessed_dataset = formatted_dataset_test.filter(
        lambda example: example['id'] not in processed_ids
    )
    
    batch_inference_llm(model=model, tokenizer=tokenizer, dataset=unprocessed_dataset, output_file_path=args.output_file)
