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
    """Process single sample - model loaded per worker (Kaggle-safe)"""
    llm = load_quantized_llm()  # Load model once per worker (Kaggle memory-friendly)
    
    prompt = f"""Please answer the question: {sample['question']}. These are the options {sample['options']}."""
    
    messages = [
        {"role": "system", "content": "You are a reasoning clinical assistant. Please carefully reason through the question and answer it."},
        {"role": "user", "content": prompt}
    ]
    
    output = llm.create_chat_completion(
        messages=messages,
        # max_tokens=128,  # Further reduced for Kaggle speed (clinical Q&A is concise)
        # temperature=0.0,
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


def parallel_processing(data, output_json_path, max_workers=1):  # 1 worker for Kaggle GPU safety
    # Load existing results (Kaggle saves to /kaggle/working/)
    if os.path.exists(output_json_path):
        with open(output_json_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
    else:
        results = []
    
    processed_ids = {r['id'] for r in results}
    to_process = [s for s in data if s['id'] not in processed_ids]
    print(f"Need to process {len(to_process)} samples")

    # Process in parallel with Kaggle constraints
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
    Worker function to fetch and format augmented context for a single ID.
    It's self-contained to be easily used by multiprocessing.
    """
    # NOTE: This function assumes that the lookup dicts and system objects
    # are available in the global scope of the worker processes.
    # This works well on Linux (like Kaggle) due to how processes are forked.
    
    try:
        # --- Safely retrieve basic info ---
        question_data = id_lookup_test_questions.get(sample_id)
        if not question_data:
            return None # Cannot proceed without a question
        
        user_question = question_data['question']
        options = question_data.get('options', {})

        # --- Retrieve PrimeKG Facts ---
        facts = []
        try:
            # We use .get() chaining for maximum safety against KeyErrors
            entities_data = id_lookup_entity.get(sample_id, {}).get('query', {}).get('entities', [])
            if entities_data:
                # The querier's method is assumed to be robust and handle bad entities
                facts = primekg_querier.get_facts_for_mcq(
                    entities=entities_data,
                    question=user_question,
                    options=options
                )
        except Exception as e:
            # This catch is for unexpected errors within the get_facts_for_mcq call
            print(f"ID {sample_id}: Error during PrimeG fact retrieval: {e}")

        # --- Consolidate and Execute RAG Queries ---

        # a) Query from entities
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
            # if queries_to_run:
                # for query in queries_to_run:
                    
        
        # --- Build a Clean Context String ---
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

        # --- Return the final dictionary ---
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
            # imap_unordered is great because it yields results as they finish,
            # which is perfect for progress bars and incremental saving.
            for result in tqdm(pool.imap_unordered(func, ids_to_process), total=len(ids_to_process)):
                if result:
                    # Write each result as a new line in the JSON Lines file
                    f_out.write(json.dumps(result) + '\n')
        
        print(f"\nProcessing complete. All new results have been saved to '{output_file_path}'.")

def batch_inference_llm(model, tokenizer, dataset, output_file_path):
    if len(dataset) == 0:
        print("All items have already been processed. Nothing to do.")
    else:
        print(f"Resuming inference for {len(dataset)} remaining items.")
        
        # --- 2. The Modified High-Throughput Inference Loop ---
        print(f"Starting batched inference with batch size {CFG.BATCH_SIZE}...")
        
        # Open the output file in 'append' mode ('a'). 
        # The 'with' statement ensures it's safely closed even if there's an error.
        with open(output_file_path, 'a') as f_out:
            
            # We iterate over the UNPROCESSED dataset now
            for i in tqdm(range(0, len(dataset), CFG.BATCH_SIZE), desc="Generating Chosen Responses"):
                # Get the batch of data, including both the text and the id
                batch = dataset[i:i + CFG.BATCH_SIZE]
                batch_prompts = batch['text']
                batch_ids = batch['id'] # Crucial for saving the correct ID with the result
                
                # Tokenize the batch (your existing code)
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
                
                # Decode the batch (your existing code, but without skip_special_tokens is safer)
                decoded_outputs = tokenizer.batch_decode(outputs)
                
                # --- On-the-Fly Saving Logic ---
                # Instead of appending to a list in memory, we write to a file on disk.
                for original_prompt, full_output, item_id in zip(batch_prompts, decoded_outputs, batch_ids):
                    # Isolate the generated part of the text
                    # Slicing from the end of the prompt is a clean way to do this
                    reasoning_and_answer = full_output[len(original_prompt):]
                    
                    # Clean up any special tokens that might be left over from generation
                    reasoning_and_answer = reasoning_and_answer.replace(tokenizer.eos_token, "").strip()

                    # Create the record we want to save
                    record = {
                        "id": item_id,
                        "chosen_response": reasoning_and_answer
                    }
                    
                    # Write the JSON object as a new line in the file and flush it
                    # to ensure it's written immediately to disk.
                    f_out.write(json.dumps(record) + '\n')
                    f_out.flush()

                # After saving outputs for the batch:
                del inputs, outputs, decoded_outputs, batch_prompts, batch_ids  # release large objects explicitly
                gc.collect()  # Python garbage collector
                torch.cuda.empty_cache()  # Free unused GPU memory

        print(f"\nInference complete. All results are saved in {output_file_path}.")