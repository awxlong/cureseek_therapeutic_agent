
import numpy as np
from llama_cpp import Llama
from sklearn.preprocessing import normalize
from typing import List, Tuple

# ==============================================================================
# Embedding Model and Helper Classes 
# ==============================================================================

class LlamaCppEmbeddings:
    """
    A wrapper class to provide a standardized interface for creating embeddings
    using a GGUF model loaded with llama-cpp-python.
    """
    def __init__(self, repo_id="second-state/gte-Qwen2-1.5B-instruct-GGUF", filename="gte-Qwen2-1.5B-instruct-Q5_K_S.gguf", **kwargs):
        """
        Initializes the Llama-based embedding model.
        
        Args:
            repo_id (str): The Hugging Face repository ID of the model.
            filename (str): The GGUF filename in the repository.
            **kwargs: Additional arguments for Llama.from_pretrained (e.g., n_gpu_layers, n_ctx).
        """
        print("Loading GGUF embedding model...")
        # Set embedding=True to use the model for embeddings
        self.llm = Llama.from_pretrained(
            repo_id=repo_id,
            filename=filename,
            embedding=True, 
            verbose=False,
            **kwargs
        )
        print("Model loaded successfully.")

    def _embed_and_average(self, text: str) -> List[float]:
        """
        Internal helper method to embed a single piece of text and return a
        single, averaged embedding vector.
        """
        # Get the raw embedding result from llama.cpp
        embedding_result = self.llm.create_embedding(text)

        # The 'data' key contains a list of dictionaries, one for each chunk
        chunk_embeddings =  embedding_result['data'][0]['embedding'] # [item['embedding'] for item in embedding_result['data'][0]['embedding']]
        
        # If the text was chunked, we get multiple embedding vectors.
        # We'll average them to get a single representative vector.
        # pdb.set_trace()
        if len(chunk_embeddings) > 1:
            # For debugging: let the user know that averaging is happening
            # print(f"    (Note: Input was split into {len(chunk_embeddings)} chunks. Averaging embeddings.)")
            avg_embedding = np.mean(np.array(chunk_embeddings), axis=0)
        else:
            avg_embedding = np.array(chunk_embeddings[0])
            
        return avg_embedding #.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generates a single, averaged embedding for each document in a list.
        """
        print(f"Generating embeddings for {len(texts)} documents...")
        # Process each text individually to handle potential chunking
        return [self._embed_and_average(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        """
        Generates a single, averaged embedding for a single query text.
        """
        return self._embed_and_average(text)



def find_best_option_by_similarity(reasoning_text: str, options: dict, embedding_model) -> Tuple[str, str]:
    """
    Finds the best multiple-choice option by comparing semantic similarity with the reasoning text.

    Args:
        reasoning_text (str): The LLM's generated reasoning.
        options (dict): The dictionary of options, e.g., {'A': 'text for A', 'B': 'text for B'}.
        embedding_model: An instance of our LlamaCppEmbeddings class.

    Returns:
        A tuple of (choice_letter, choice_text) for the best match, or None if options are invalid.
    """
    if not reasoning_text or not options or not isinstance(options, dict):
        return None

    option_texts = list(options.values())
    option_keys = list(options.keys())

    # 1. Embed the reasoning text (our query)
    reasoning_embedding = np.array(embedding_model.embed_query(reasoning_text)).reshape(1, -1)
    # pdb.set_trace()
    # 2. Embed all the option texts (our documents)
    option_embeddings = np.array(embedding_model.embed_documents(option_texts))

    # 3. Normalize all embeddings for cosine similarity calculation
    reasoning_embedding_norm = normalize(reasoning_embedding, axis=1, norm='l2')
    option_embeddings_norm = normalize(option_embeddings, axis=1, norm='l2')

    # 4. Calculate cosine similarity
    similarities = np.dot(reasoning_embedding_norm, option_embeddings_norm.T)[0]

    # 5. Find the index of the best-matching option
    best_option_index = np.argmax(similarities)

    # 6. Return the corresponding choice letter and text
    best_choice_letter = option_keys[best_option_index]
    best_choice_text = option_texts[best_option_index]
    
    return best_choice_letter, best_choice_text