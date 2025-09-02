"""
Code for performing inference with LLMs on Kaggle's cloud environment

"""
import multiprocessing
from tqdm import tqdm
from llm import load_quantized_llm
import os
from functools import partial
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc 
import torch 
from CFG import CFG

def process_sample(sample):
    """Process a single question-answer sample using a quantized LLM
    Loads model once per worker (memory-efficient for constrained environments like Kaggle)
    Generates a reasoning-based response to a clinical question with provided options
    Args:
        sample: dict - entry from json file
    Returns:
        dict - structured result with input metadata and model output
    """
    llm = load_quantized_llm()  # Load model once per worker (Kaggle memory-friendly)
    
    prompt = f"""Please answer the question: {sample['question']}. These are the options {sample['options']}."""
    
    messages = [
        {"role": "system", "content": "You are a reasoning clinical assistant. Please carefully reason through the question and answer it."},
        {"role": "user", "content": prompt}
    ]
    
    output = llm.create_chat_completion(
        messages=messages,
        stream=False,  # Disable streaming (faster in Kaggle)
    )
    
    return {
        'id': sample['id'],
        'question_type': sample['question_type'],
        'question': sample['question'],
        'options': sample['options'],
        'llm_answer': output['choices'][0]['message']['content'],
        'output': str(output)
    }


def parallel_processing(data, output_json_path, max_workers=1): 
    """
    Process multiple samples in parallel with checkpointing
    Resumes from existing results file if available (avoids reprocessing)
    Uses process pooling for parallel execution while respecting resource constraints
    Saves results incrementally to handle potential interruptions
    Args:
        data: dict - entry from json file
        output_json_path: str - name of (partial) output file, which is checked whether
        it already exists to resume from previous incomplete run
        max_worker: int - number of threads in multiprocessing
    Returns:
        dict - output file with answers
    """
    # Load existing results (Kaggle saves to /kaggle/working/)
    if os.path.exists(output_json_path):
        with open(output_json_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
    else:
        results = []
    
    # Identify unprocessed samples using their unique IDs
    processed_ids = {r['id'] for r in results}
    to_process = [s for s in data if s['id'] not in processed_ids]
    print(f"Need to process {len(to_process)} samples")

    # Process in parallel with Kaggle's computational constraints
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_sample, sample): sample for sample in to_process}

        for future in as_completed(futures):
            sample = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(f"Processed sample {result['id']}")

                # Save to Kaggle working directory (persists between sessions)
                with open(output_json_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=4, ensure_ascii=False)
            except Exception as e:
                print(f"Error processing sample {sample['id']}: {e}")
                # Save error to results for debugging
                results.append({
                    'id': sample['id'],
                    'error': str(e)
                })
                with open(output_json_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=4, ensure_ascii=False)


def process_context_augmentation(sample_id, id_lookup_test_questions, id_lookup_entity, primekg_querier, rag_system):
    """
    Worker function to create augmented context for a single (test) question
    Combines knowledge graph facts (PrimeKG) and relevant literature passages (via EuropeanPMC)
    Designed for safe multiprocessing execution with robust error handling later
    Args:
        sample_id: str - unique 'id' of the question
        id_lookup_test_questions: dict - json file which enables checking corresponding test entry given id
        id_lookup_entity: dict - json file with entities to construct a query, e.g., "juvenile rheumatoid arthritis",
                "CYP2C9", etc.
        primekg_querier: obj - performs queries over PrimeKG given query
        rag_system: obj - performs search over European PMC given complex queries
    Returns:
        dict - structured context data or None if processing fails
    """
    
    try:
        # --- Safely retrieve basic info ---
        question_data = id_lookup_test_questions.get(sample_id)
        if not question_data:
            return None 
        
        user_question = question_data['question']
        options = question_data.get('options', {})

        # --- Retrieve PrimeKG Facts ---
        facts = []
        try:
            entities_data = id_lookup_entity.get(sample_id, {}).get('query', {}).get('entities', [])
            if entities_data:
                facts = primekg_querier.get_facts_for_mcq(
                    entities=entities_data,
                    question=user_question,
                    options=options
                )
        except Exception as e:
            print(f"ID {sample_id}: Error during PrimeG fact retrieval: {e}")

        # ---  Query European PMC ---
        entity_query_list = id_lookup_entity.get(sample_id, {}).get('query', {}).get('entities', [])
        entity_query_list = [s for s in entity_query_list if not ("none" in s.lower())]
        if entity_query_list:

            all_passages = []
            seen_passage_texts = set()

            # loop through query until a PMC search works
            range_query_list = entity_query_list.copy()
            for negative_idx in range(0, len(range_query_list)):
                entity_query_list = range_query_list[:len(range_query_list) - negative_idx]
                query = str(" AND ".join(entity_query_list))
                if len(all_passages) == 0:
                    print(f'Using query: {query}')
                    try:
                        passages = rag_system.search_and_rank_passages(
                            query=query,
                            user_question=user_question,
                            max_articles=1,
                            max_passages_per_llm_context=1,
                            rank=True
                        )
                        for passage in passages:
                            if passage['text'] not in seen_passage_texts:
                                all_passages.append(passage)
                                seen_passage_texts.add(passage['text'])
                    except Exception as e:
                        print(f"ID {sample_id}: Error during RAG for query '{query[:50]}...': {e}")
                elif len(all_passages) != 0:
                    break
                
        # --- Build a Clean Augmented Context ---
        context_parts = []
        if facts:
            context_parts.append("[PRIMEKG KNOWLEDGE GRAPH FACTS]\n" + "\n".join([f"- {fact}" for fact in facts]))
        if all_passages:
            passage_section = "[RELEVANT PASSAGES FROM LITERATURE]\n"
            for i, p in enumerate(all_passages, 1):
                passage_section += f"--- Passage {i} from Article ID: {p['article_id']} ---\n{p['text']}\n"
            context_parts.append(passage_section)
        
        if not context_parts:
            augmented_context = "No additional context could be found."
        else:
            augmented_context = "\n\n".join(context_parts)

        return {
            "id": sample_id,
            "augmented_context": augmented_context
        }

    except Exception as e:
        print(f"FATAL ERROR processing ID {sample_id}: {e}")
        return None

def parallel_processing_context_augmentation(ids_to_process, all_incomplete_ids, processed_ids, output_file_path, \
                                             id_lookup_test_questions, id_lookup_entity,\
                                             primekg_querier, rag_system):
    
    """
    Parallel processing of process_context_augmentation to loop over entire test set with
    resumability logic. 
    Args: 
        same as process_context_augmentation()
    Returns:
        dict - json file with augmented context 
    
    """
    if not ids_to_process:
        print("All entries have already been processed. Nothing to do.")
    else:
        print(f"Total incomplete: {len(all_incomplete_ids)}. Already processed: {len(processed_ids)}. To be processed now: {len(ids_to_process)}")

        # --- MULTIPROCESSING EXECUTION ---
        func = partial(
            process_context_augmentation, 
            id_lookup_test_questions=id_lookup_test_questions, 
            id_lookup_entity=id_lookup_entity, 
            primekg_querier=primekg_querier, 
            rag_system=rag_system
        )

        num_processes = 2
        print(f"Starting multiprocessing pool with {num_processes} workers...")

        # Open the file in append mode to add new results
        with open(output_file_path, 'a') as f_out, multiprocessing.Pool(processes=num_processes) as pool:

            # use imap_unordered to enable progress bars and incremental saving.
            for result in tqdm(pool.imap_unordered(func, ids_to_process), total=len(ids_to_process)):
                if result:
                    # Write each result as a new line in the JSON Lines file
                    f_out.write(json.dumps(result) + '\n')
        
        print(f"\nProcessing complete. All new results have been saved to '{output_file_path}'.")

def batch_inference_llm(model, tokenizer, dataset, output_file_path):
    """
    High-throughput batched LLM inference with memory management
    Processes multiple prompts simultaneously for efficiency
    Saves results incrementally to prevent memory overload
    Implements explicit memory cleanup for GPU resources
    Uses configuration from CFG for batch size and generation parameters
    Args:
        model: obj - LLM (base or finetuned)
        tokenizer: obj - LLM's tokenizer
        dataset: obj - Torch dataset with chat template applied
        output_file_path: str - name of file to save (partial) results which
            is checked for resumability
    Returns:
        dict - (partial) file with test answers
    
    """
    if len(dataset) == 0:
        print("All items have already been processed. Nothing to do.")
    else:
        print(f"Resuming inference for {len(dataset)} remaining items.")
        
        # --- Batch Inference Loop ---
        print(f"Starting batched inference with batch size {CFG.BATCH_SIZE}...")
        
        with open(output_file_path, 'a') as f_out:
            
            # We iterate over the UNPROCESSED dataset only
            for i in tqdm(range(0, len(dataset), CFG.BATCH_SIZE), desc="Generating Chosen Responses"):
                
                batch = dataset[i:i + CFG.BATCH_SIZE]
                batch_prompts = batch['text']
                batch_ids = batch['id'] # Crucial for saving the correct ID with the result
         
                inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model.device)
                
                # Generate responses for the entire batch (your existing code)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=True,
                    temperature=0.32,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )
                decoded_outputs = tokenizer.batch_decode(outputs)
                
                # --- On-the-Fly Saving Logic to Resume Later ---
                for original_prompt, full_output, item_id in zip(batch_prompts, decoded_outputs, batch_ids):
                    # Isolate the generated part of the text
                    reasoning_and_answer = full_output[len(original_prompt):]
                    
                    # Clean up any special tokens that might be left over from generation
                    reasoning_and_answer = reasoning_and_answer.replace(tokenizer.eos_token, "").strip()

                    record = {
                        "id": item_id,
                        "chosen_response": reasoning_and_answer
                    }
                    
                    f_out.write(json.dumps(record) + '\n')
                    f_out.flush()

                # Clean memory
                del inputs, outputs, decoded_outputs, batch_prompts, batch_ids  # release large objects explicitly
                gc.collect()  # Python garbage collector
                torch.cuda.empty_cache()  # Free unused GPU memory

        print(f"\nInference complete. All results are saved in {output_file_path}.")