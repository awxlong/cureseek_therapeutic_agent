import os
import wandb
from datasets import load_dataset
from args import get_args_sft_deepseek
from dataset import format_dataset_entry_for_sft
from finetune import supervised_finetune
from llm import load_llm
import transformers

transformers.set_seed(42)

wandb.login(key=os.getenv('WANDB_LOGIN_KEY'))

if __name__=="__main__":
    args = get_args_sft_deepseek()

    original_model, tokenizer = load_llm(model_path=args.model_path, quantized=args.quantized)

    print("Loading raw dataset...")
    dataset = load_dataset('json', data_files=args.data_file_path, split='train')
    print(f"Original dataset size: {len(dataset)}")

    sft_dataset = dataset.filter(lambda example: example['correct_or_not'])
    print(f"Filtered dataset size (correct answers only): {len(sft_dataset)}")

    formatted_dataset = sft_dataset.map(lambda example: format_dataset_entry_for_sft(example, tokenizer))

    print("\n--- Example of Correctly Formatted Text for Training ---")
    print(formatted_dataset[-1]['text'])
    print("-------------------------------------------------------\n")

    supervised_finetune(model=original_model, tokenizer=tokenizer, dataset=formatted_dataset)