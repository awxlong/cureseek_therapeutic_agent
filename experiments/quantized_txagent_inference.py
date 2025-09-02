"""
Code for running inference with quantized TxAgent on test questions. 
TxAgent is described in https://github.com/mims-harvard/TxAgent
"""
import json
from inference import parallel_processing
from args import get_args_quantized_txagent_inference


if __name__ == "__main__":

    args = get_args_quantized_txagent_inference()

    # Loading test set
    with open(args.data_path, 'r') as f:
        data = [json.loads(line) for line in f]

    # Resumability: Load IDs that have already been processed 
    if args.resume:
        with open(args.resume_file, 'r') as f:
            temp_data = json.load(f)
        with open(args.output_path, 'w', encoding='utf-8') as f:
            json.dump(temp_data, f, indent=4, ensure_ascii=False)

    parallel_processing(data, args.output_path, max_workers=args.max_workers)
