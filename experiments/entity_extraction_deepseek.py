import os
from datasets import load_dataset
from args import get_args_entity_extraction_deepseek
from dataset import format_dataset_entry_entity_extraction
from inference import batch_inference_llm
from llm import load_llm
import transformers
import torch
import json 

transformers.set_seed(42)

if __name__=="__main__":
    args = get_args_entity_extraction_deepseek()
    
    print("Loading and setting LLM to evaluation mode.")
    original_model, tokenizer = load_llm(model_path=args.model_path, quantized=args.quantized)
    original_model.eval()
    print("Compiling the model for faster inference... (this may take a minute)")
    # Use `mode="reduce-overhead"` for generation tasks
    compiled_model = torch.compile(original_model, mode="reduce-overhead")
    print("Model compiled successfully.")

    print("Loading test dataset...")
    dataset = load_dataset('json', data_files=args.data_file_path, split='train')
    print(f"Original dataset size: {len(dataset)}")

    entity_generation_dataset = dataset.map(lambda example: format_dataset_entry_entity_extraction(example, tokenizer))

    print("\n--- Example of Entity Extraction Dataset Entry ---")
    print(entity_generation_dataset[-1]['text'])
    print("-------------------------------------------------------\n")

    # ** RESUMABILITY: Load IDs that have already been processed **
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
    unprocessed_dataset = entity_generation_dataset.filter(
        lambda example: example['id'] not in processed_ids
    )
        
    batch_inference_llm(model=compiled_model, tokenizer=tokenizer, dataset=unprocessed_dataset, output_file_path=args.output_file)

