"""
Code for running inference with quantized TxAgent on test questions via RAG
using context augmented thanks to information extracted from European PMC and
PrimeKG. 
TxAgent is described in https://github.com/mims-harvard/TxAgent
"""

from RAG import MedRAG, PrimeKG_Querier
from args import get_args_context_augmentation
from inference import parallel_processing_context_augmentation
from semantic_search import LlamaCppEmbeddings
import json
import pandas as pd
import os

if __name__=="__main__":

    args = get_args_context_augmentation()

    # Loading necessary datasets
    with open(args.entity_extraction_path, 'r') as f:
        deepseek_entity_extraction = json.load(f)
    with open(args.complex_query_path, 'r') as f:
        deepseek_complex_query = json.load(f)
    with open(args.test_questions_path, 'r') as f:
        test_questions = [json.loads(l) for l in f]
    
    # Loading test set with partial answers, i.e., No answer entries to be filled in
    incomplete_quantized_txagent = pd.read_csv(args.incomplete_submission)

    # Loading retrieval systems
    embedding_model = LlamaCppEmbeddings(
        repo_id=args.embedding_repo_id,
        filename=args.embedding_filename,
    )
    rag_system = MedRAG(embedding_model=embedding_model, email=args.medrag_email)
    primekg_querier = PrimeKG_Querier(kg_path=args.kg_path)

    id_lookup_query = {sample['id']: sample for sample in deepseek_complex_query}
    id_lookup_entity = {sample['id']: sample for sample in deepseek_entity_extraction}
    id_lookup_test_questions = {sample['id']: sample for sample in test_questions}

    # Resumability: Load IDs that have already been processed 
    with open(args.progress_path, 'r') as f:
        temp = [json.loads(l) for l in f]
        
    with open(args.output_file, 'w') as f:
        for obj in temp:
            f.write(json.dumps(obj) + '\n')

    processed_ids = set()
    if os.path.exists(os.path.join('/kaggle/working/', args.output_file)):
        print(f"Output file '{args.output_file}' found. Loading previously processed IDs...")
        with open(os.path.join('/kaggle/working/', args.output_file), 'r') as f:
            for line in f:
                data = json.loads(line)
                # print(data['id'])
                processed_ids.add(data['id'])

    print(f"Found {len(processed_ids)} completed entries. Resuming from where we left off.")
    all_incomplete_ids = set(incomplete_quantized_txagent[incomplete_quantized_txagent['choice'] == 'No answer']['id'])
    
    ids_to_process = sorted(list(all_incomplete_ids - processed_ids)) # Process only the ones not yet done

    parallel_processing_context_augmentation(ids_to_process=ids_to_process, all_incomplete_ids=all_incomplete_ids, processed_ids=processed_ids, output_file_path=args.output_file, \
                                             id_lookup_test_questions=id_lookup_test_questions, id_lookup_entity=id_lookup_entity,\
                                             primekg_querier=primekg_querier, rag_system=rag_system)



