"""
Helper functions doing prompt engineering and applying 
chat template to perform inference on LLMs
"""

def format_dataset_entry_self_correction(example, tokenizer):
    """
    Applies the model's chat template with a system prompt.

    Args:
        example: dict - sample from json file
        tokenizer: obj - LLM tokenizer

    Returns:
        dict - sample text with chat template applied
    """
    SELF_CORRECTION_SYSTEM_PROMPT = """You are an expert clinical reasoning agent specializing in pharmaceutical therapeutics. Your task is to analyze multiple-choice questions. First, provide a detailed step-by-step reasoning process within <think> tags, explaining why the correct option is right and the others are wrong. Then, state the final correct answer in the format 'Letter: Full Answer Text'."""

    user_content = f"{example['question']} \nOptions: {example['options']} \nThe correct final answer is known to be: '{example['correct_answer']}'. Please provide the reasoning or the tools that would help you infer the correct answer. "
    
    messages = [
        {"role": "system", "content": SELF_CORRECTION_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    
    formatted_text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    return {"text": formatted_text}


def format_dataset_entry_for_sft(example, tokenizer):
    """
    Applies the model's chat template with a system prompt.

    Args:
        example: dict - sample from json file
        tokenizer: obj - LLM tokenizer

    Returns:
        dict - sample text with chat template applied
    """

    user_content = example['question']
    assistant_content = f"<think>{example['reasoning']}</think> The correct answer is {example['correct_answer']}"
    
    messages = [
        {"role": "system", "content": """You are an expert clinical reasoning agent specializing in pharmaceutical therapeutics. Your task is to analyze multiple-choice questions. First, provide a detailed step-by-step reasoning process within <think> tags, explaining why the correct option is right and the others are wrong. Then, state the final correct answer in the format 'Letter: Full Answer Text'."""},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content}
    ]
    
    formatted_text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=False
    )
    
    return {"text": formatted_text}


def format_dataset_entry_for_inference(example, tokenizer):
    """
    Applies the model's chat template with a system prompt.

    Args:
        example: dict - sample from json file
        tokenizer: obj - LLM tokenizer

    Returns:
        dict - sample text with chat template applied
    """

    user_content = f"{example['question']} \nOptions: {example['options']} \nPlease carefully reason through the question and infer the correct answer."
    
    GENERATION_SYSTEM_PROMPT = """You are an expert clinical reasoning agent specializing in pharmaceutical therapeutics. Your task is to analyze multiple-choice questions. Provide a reasoning process to infer the correct answer and why other options are wrong. Then, output the answer in the format 'Letter: Full Answer Text'."""
    
    messages = [
        {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    
    formatted_text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    return {"text": formatted_text}


def format_dataset_entry_entity_extraction(example, tokenizer):
    """
    Applies the model's chat template with a system prompt.

    Args:
        example: dict - sample from json file
        tokenizer: obj - LLM tokenizer

    Returns:
         dict - sample text with chat template applied
    """

    GENERATION_SYSTEM_PROMPT = """ You are a biomedical entity recognition specialist. Your only task is to identify all relevant drugs, diseases, genes, and proteins from the provided question and options, NOT to answer it.
    **Instructions:**
    - Identify all potential biomedical entities.
    - Do not invent or infer entities that are not present in the text.
    - Respond ONLY with a single JSON object containing a list of strings called "entities".
    ---
    **EXAMPLE**

    **Question:**
    Which of the following is a potential pharmacogenetic consideration for a patient with juvenile rheumatoid arthritis being treated with a drug metabolized by CYP2C9?

    **Options:**
    A) Increased risk of toxicity in CYP2D6 poor metabolizers.
    B) Variants in the CYP2C9 gene may affect drug efficacy and safety.
    C) HLA-B*57:01 allele is associated with hypersensitivity reactions.
    D) Thiopurine methyltransferase (TPMT) deficiency leads to reduced drug clearance.

    **Your Response:**
    {{
    "entities": [
        "juvenile rheumatoid arthritis",
        "CYP2C9",
        "CYP2D6",
        "HLA-B*57:01",
        "Thiopurine methyltransferase",
        "TPMT"
    ]
    }}
    ---

    """

    user_content = f"{example['question']} \nOptions: {example['options']}. \nPlease extract relevant biomedical entities in JSON format. "
    
    messages = [
        {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    
    formatted_text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    return {"text": formatted_text}