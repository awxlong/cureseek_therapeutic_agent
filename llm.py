"""
Code is mainly elaborated upon: https://www.kaggle.com/code/danielphalen/grpotrainer-deepseekr1
"""


from llama_cpp import Llama
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
) 

from CFG import *

import torch


def load_llm(model_path='/kaggle/input/deepseek-r1/transformers/deepseek-r1-distill-qwen-1.5b/2', quantized=True):
    if quantized:
        compute_dtype = getattr(torch, "float16")
        bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=False,
            )
        original_model = AutoModelForCausalLM.from_pretrained(model_path, 
                                                            device_map=device_map,
                                                            quantization_config=bnb_config,
                                                            trust_remote_code=True,
                                                            attn_implementation=attn_implementation # Apply the selected attention mechanism
                                                            )
        tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                trust_remote_code=True,
                                                # padding_side="left"
                                                )
    else:
        original_model = AutoModelForCausalLM.from_pretrained(model_path, 
                                                            device_map=device_map,
                                                            trust_remote_code=True,
                                                            attn_implementation=attn_implementation # Apply the selected attention mechanism
                                                            )
        tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                trust_remote_code=True,
                                                # padding_side="left"
                                                )

    # --- ENSURING THE CHAT TEMPLATE ENABLES ADDING <think></think> tags---
    original_template = tokenizer.chat_template
    str_to_remove = "{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}"

    if str_to_remove in original_template:
        print("Found problematic logic in chat template. Removing it...")
        corrected_template = original_template.replace(str_to_remove, "")
        tokenizer.chat_template = corrected_template
        print("Chat template has been corrected in-memory for this session.")
    else:
        print("Chat template does not contain the problematic logic. Proceeding as is.")

    return original_model, tokenizer


def load_finetuned_llm(model_path='kaggle/working/cureseek-v2-merged', quantized=True):
    if quantized:
        compute_dtype = getattr(torch, "float16")
        bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=False,
            )
        original_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation=attn_implementation
            # trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, 
                                          trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token 

    else:
        original_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation=attn_implementation
            # trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, 
                                          trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token 

    # --- ENSURING THE CHAT TEMPLATE ENABLES ADDING <think></think> tags---
    original_template = tokenizer.chat_template
    str_to_remove = "{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}"

    if str_to_remove in original_template:
        print("Found problematic logic in chat template. Removing it...")
        corrected_template = original_template.replace(str_to_remove, "")
        tokenizer.chat_template = corrected_template
        print("Chat template has been corrected in-memory for this session.")
    else:
        print("Chat template does not contain the problematic logic. Proceeding as is.")

    return original_model, tokenizer


def load_quantized_llm(repo_id = "mradermacher/TxAgent-T1-Llama-3.1-8B-GGUF", filename = "TxAgent-T1-Llama-3.1-8B.Q8_0.gguf"):
    # """Load model with Kaggle GPU constraints"""
    # os.environ["GGML_CUDA_FORCE_MMQ"] = "1"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Ensure Kaggle's GPU is visible
    
    llm = Llama.from_pretrained(
        repo_id=repo_id,
        filename=filename,
        # n_ctx=2048, # default 512
        n_gpu_layers=35,  # Kaggle GPUs (T4/V100) work best with ~35 layers (avoids OOM)
        n_threads=2,  # Kaggle has 4 CPU cores - limit per-worker threads
        n_batch=4096,  # 
        verbose=False,  # Reduce log spam in Kaggle console
        flash_attn=True,
        # device='cuda'
    )

    return llm
