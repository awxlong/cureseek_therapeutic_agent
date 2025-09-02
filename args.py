"""
Code for parsing arguments for files in experiments/
"""

import argparse

def get_args_deepseek_inference():
    parser = argparse.ArgumentParser(description="Arguments for loading and running LLM inference")

    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to the model directory or file."
    )
    parser.add_argument(
        "--quantized", action="store_true",
        help="Use this flag to load a quantized version of the model."
    )
    parser.add_argument(
        "--data_file_path", type=str, required=True,
        help="Path to the JSON file containing the test dataset."
    )
    parser.add_argument(
        "--output_file", type=str, default="test_results_original_deepseek.jsonl",
        help="Path to save output results. Default is 'test_results_original_deepseek.jsonl'"
    )
    parser.add_argument(
        "--load_finetuned", action="store_true",
        help="Set this flag to load the fine-tuned merged model (overrides load_finetuned=True in code)."
    )

    return parser.parse_args()

def get_args_quantized_txagent_inference():
    parser = argparse.ArgumentParser(description="Process CureBench data with optional resume capability.")

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from a previous results file"
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default="/kaggle/input/curebench-data/curebench_testset_phase1.jsonl",
        help="Path to the input JSONL data file"
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="/kaggle/working/curebench_results.json",
        help="Path to save the output results JSON"
    )

    parser.add_argument(
        "--resume_file",
        type=str,
        help="Path to the resume JSON file containing previous results"
    )

    parser.add_argument(
        "--max_workers",
        type=int,
        default=2,
        help="Number of parallel workers for processing"
    )

    args = parser.parse_args()
    return args


def get_args_sft_deepseek():
    parser = argparse.ArgumentParser(description="Fine-tune LLM on custom dataset")

    parser.add_argument(
        "--model_path",
        type=str,
        default="/kaggle/input/deepseek-r1/transformers/deepseek-r1-distill-qwen-1.5b/2",
        help="Path to the pre-trained LLM model"
    )

    parser.add_argument(
        "--quantized",
        action="store_true",
        help="Load the model in quantized mode"
    )

    parser.add_argument(
        "--data_file_path",
        type=str,
        default="/kaggle/input/curebench-data/curebench_testset_phase1.jsonl",
        required=True,
        help="Path to the JSON dataset file for training"
    )

    args = parser.parse_args()
    return args

def get_args_entity_extraction_deepseek():
    parser = argparse.ArgumentParser(description="Fine-tune LLM on custom dataset")

    parser.add_argument(
        "--model_path",
        type=str,
        default="/kaggle/input/deepseek-r1/transformers/deepseek-r1-distill-qwen-1.5b/2",
        help="Path to the pre-trained LLM model"
    )

    parser.add_argument(
        "--quantized",
        action="store_true",
        help="Load the model in quantized mode"
    )

    parser.add_argument(
        "--data_file_path",
        type=str,
        default="/kaggle/input/curebench-data/curebench_testset_phase1.jsonl",
        required=True,
        help="Path to the JSON dataset file for training"
    )

    parser.add_argument(
        "--output_file",
        type=str,
        default="entity_extraction.jsonl",
        help="Output JSONL file path for writing entity recognitition file"
    )

    args = parser.parse_args()
    return args

def get_args_self_correction_deepseek():
    parser = argparse.ArgumentParser(description="Fine-tune LLM on custom dataset")

    parser.add_argument(
        "--model_path",
        type=str,
        default="/kaggle/input/deepseek-r1/transformers/deepseek-r1-distill-qwen-1.5b/2",
        help="Path to the pre-trained LLM model"
    )

    parser.add_argument(
        "--quantized",
        action="store_true",
        help="Load the model in quantized mode"
    )

    parser.add_argument(
        "--data_file_path",
        type=str,
        default="/kaggle/input/curebench-data/curebench_testset_phase1.jsonl",
        required=True,
        help="Path to the JSON dataset file for training"
    )

    parser.add_argument(
        "--output_file",
        type=str,
        default="self_correct_partial_results.jsonl",
        help="Output JSONL file path for writing self-correction files"
    )

    args = parser.parse_args()
    return args

def get_args_context_augmentation():

    parser = argparse.ArgumentParser(description="Process CureBench data for context augmentation to perform RAG.")

    # Data file paths
    parser.add_argument(
        "--entity_extraction_path",
        type=str,
        default="/kaggle/input/curebench-rag-data/entity_extraction_processed.json",
        help="Path to the deepseek entity extraction JSON file"
    )
    parser.add_argument(
        "--complex_query_path",
        type=str,
        default="/kaggle/input/curebench-rag-data/complex_search_queries_processed.json",
        help="Path to the deepseek complex search queries JSON file"
    )
    parser.add_argument(
        "--test_questions_path",
        type=str,
        default="/kaggle/input/curebench-data/curebench_testset_phase1.jsonl",
        help="Path to the test questions JSONL file"
    )
    parser.add_argument(
        "--incomplete_submission",
        type=str,
        default="/kaggle/input/curebench-rag-data/test_submission_beta.csv",
        help="Path to submission CSV file with No answer entries"
    )
    parser.add_argument(
        "--progress_path",
        type=str,
        # default="/kaggle/input/progress-context-augmentation-3/context_augmentation.jsonl",
        help="Path to the progress file for resuming context augmentation"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="rag_inference.jsonl",
        help="Output JSONL file path for writing augmented contexts"
    )

    # MedRAG / RAG system specific configs
    parser.add_argument(
        "--kg_path",
        type=str,
        default="/kaggle/input/curebench-rag-data/kg.csv",
        help="Path to the knowledge graph CSV file for PrimeKG querier"
    )
    parser.add_argument(
        "--embedding_repo_id",
        type=str,
        default="second-state/gte-Qwen2-1.5B-instruct-GGUF",
        help="Repository ID for the embedding model"
    )
    parser.add_argument(
        "--embedding_filename",
        type=str,
        default="gte-Qwen2-1.5B-instruct-Q5_K_S.gguf",
        help="Filename of the embedding model weight file"
    )
    parser.add_argument(
        "--medrag_email",
        type=str,
        default="loremipsum@mail.com",
        help="Email identifier for MedRAG system initialization"
    )

    args = parser.parse_args()
    return args



