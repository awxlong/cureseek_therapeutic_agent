"""
Code containing reusable hyperparameters of used LLMs.
"""

import torch

class CFG:
    MAX_TRAIN = 100
    MAX_TOKENS = 2048
    NUM_GENERATIONS = 4
    USE_PEFT = True
    BATCH_SIZE=1
    MAX_STEPS = 80
    
    BETA = 0.04
    LR = 1.e-5
    
    MODEL_NAME = '/kaggle/input/deepseek-r1/transformers/deepseek-r1-distill-qwen-1.5b/2'
    
    step_count=10
    DEBUG = False

    SYSTEM_PROMPT = """You are an expert clinical reasoning agent specializing in pharmaceutical therapeutics. Your task is to analyze multiple-choice questions. First, provide a detailed step-by-step reasoning process within <think> tags, explaining why the correct option is right and the others are wrong. Then, state the final correct answer in the format 'Letter: Full Answer Text'."""

    DATA_FILE_PATH = "/kaggle/input/curebench-full-sft/curebench_validation_preprocessed_self_correct.json" 
    OUTPUT_DIR = "cureseek-sft-full-v2"

# --- Training Arguments ---
compute_dtype = getattr(torch, "bfloat16")
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    print("Using bfloat16 for training.")
    fp16_enabled = False
    bf16_enabled = True
    print("GPU supports Flash Attention 2. Using it for faster training.")
    attn_implementation = "flash_attention_2"
    use_packing = True
else:
    print("GPU does not support bfloat16. Using fp16.")
    fp16_enabled = True
    bf16_enabled = False
    print("GPU does not support Flash Attention 2. Using standard attention and disabling packing.")
    attn_implementation = "sdpa" # "sdpa" is the standard PyTorch attention
    use_packing = False

device_map = 'auto'