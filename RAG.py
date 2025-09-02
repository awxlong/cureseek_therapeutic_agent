#!/usr/bin/env python3
"""
Classes and helper functions for Retrieval Augmented Generation (RAG)
"""

import requests
import xml.etree.ElementTree as ET
import time
import re
from typing import List
import json
import numpy as np
from sklearn.preprocessing import normalize
import pandas as pd
import pdb

class PrimeKG_Querier:
    """
    A class to intelligently query the PrimeKG knowledge graph by using an LLM
    for entity extraction and robust Python code for the actual query logic.
    """
    def __init__(self, kg_path: str):
        """
        Loads the PrimeKG DataFrame and stores the LLM instance.

        Args:
            kg_path (str): The file path to 'kg.csv'.
            llm_instance: Your initialized deepseek-r1-distill model.
        """

        print(f"Loading PrimeKG from {kg_path}...")
        try:
            self.kg = pd.read_csv(kg_path, low_memory=True)
            # Create a pre-processed set of all unique entity names for fast validation
            all_entities = pd.concat([self.kg['x_name'], self.kg['y_name']]).dropna().unique()
            self.known_entity_set = {name.lower() for name in all_entities}
            print(f"PrimeKG loaded successfully with {len(self.known_entity_set)} unique entities.")
        except FileNotFoundError:
            print(f"ERROR: PrimeKG file not found at '{kg_path}'. This module will be disabled.")
            self.kg = None

    def get_facts_for_mcq(self, entities: dict, question: str, options: dict, max_facts: int = 20) -> List[str]:
        """
        The main public method to get facts for a given MCQ.

        Returns:
            A list of formatted strings representing the facts.
        """
        if self.kg is None:
            return []

        # Step 1: Use LLM to get a list of potential entities
        potential_entities = entities
        if not potential_entities:
            print("No potential entities extracted by the LLM.")
            return []
        print(f"\nLLM extracted {len(potential_entities)} potential entities.")

        # Step 2: Validate entities against the KG (Mitigates Hallucination)
        # This is the key step for robustness!
        valid_entities = [entity for entity in potential_entities if entity in self.known_entity_set]
        print(f"Found {len(valid_entities)} valid entities that exist in PrimeKG: {valid_entities}")

        if not valid_entities:
            return []

        # Step 3: Execute the fixed, reliable pandas query
        # We query for facts where the entity is either the subject (x_name) or object (y_name)
        facts_df = self.kg[
            self.kg['x_name'].str.lower().isin(valid_entities) |
            self.kg['y_name'].str.lower().isin(valid_entities)
        ]
        
        formatted_facts = []
        for _, row in facts_df.head(max_facts).iterrows():
            fact_str = f"{row['x_name']} -[{row['display_relation']}]-> {row['y_name']}"
            formatted_facts.append(fact_str)
        
        print(f"\nRetrieved {len(formatted_facts)} facts from PrimeKG.")
        return formatted_facts
    
class MedRAG:
    """
    A unified RAG system that intelligently searches both peer-reviewed literature
    and pre-prints via the Europe PMC and NCBI APIs.
    """
    def __init__(self, embedding_model, tool_name="MedRAG_System", email="your.email@example.com"):
        self.europe_pmc_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
        self.ncbi_base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.id_converter_url = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
        self.tool_name = tool_name
        self.email = email
        self.embedding_model = embedding_model  # <-- CRITICAL: The class now holds the embedding model
        if not self.embedding_model:
            raise ValueError("An embedding model instance is required for UnifiedRAG.")
        print("MedRAG system initialized with embedding-based passage ranking.")

    # --- Private helper methods for fetching and chunking ---
    
    def _chunk_text(self, text: str, chunk_size: int = 256, overlap: int = 32) -> List[str]:
        """Splits text into smaller, overlapping chunks based on word count."""
        words = text.split()
        if not words: return []
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks

    def _convert_pmids_to_pmcids(self, pmids: List[str]) -> dict:
        if not pmids: return {}
        params = {"ids": ",".join(pmids), "format": "json", "tool": self.tool_name, "email": self.email}
        try:
            response = requests.get(self.id_converter_url, params=params)
            response.raise_for_status()
            id_map = {record["pmid"]: record["pmcid"] for record in response.json().get("records", []) if "pmcid" in record}
            print(f"ID Converter: Found {len(id_map)} PMCIDs for {len(pmids)} PMIDs.")
            return id_map
        except (requests.exceptions.RequestException, json.JSONDecodeError):
            return {}

    def _parse_best_available_content(self, article_xml: ET.Element) -> str:
        """
        Parses an XML element for an article and returns the best available text
        in a hierarchical order: Body > Abstract > Title.

        This is the key to robustly handling different article types and access levels.
        """
        # Priority 1: Try to find the full body text.
        body_node = article_xml.find('.//body')
        if body_node is not None:
            # Use itertext() within the body to get all text from its children
            body_text = " ".join(body_node.itertext())
            # Check if the extracted text is substantial and not just whitespace
            if len(body_text.strip()) > 100: # Heuristic for non-empty body
                return re.sub(r'\s+', ' ', body_text).strip()

        # Priority 2: If no body, find the abstract.
        # Note: PubMed and PMC use different tag names for the abstract.
        abstract_node = article_xml.find('.//abstract')  # For PMC
        if abstract_node is None:
            abstract_node = article_xml.find('.//AbstractText') # For PubMed
        
        if abstract_node is not None:
            abstract_text = " ".join(abstract_node.itertext())
            return re.sub(r'\s+', ' ', abstract_text).strip()

        # Priority 3: If all else fails, return the title.
        title_node = article_xml.find('.//article-title') # For PMC
        if title_node is None:
            title_node = article_xml.find('.//ArticleTitle') # For PubMed

        if title_node is not None:
            title_text = " ".join(title_node.itertext())
            return re.sub(r'\s+', ' ', title_text).strip()
        
        # Last resort
        return "No content available."


    def _fetch_full_text_from_pmc(self, pmcids: List[str]) -> List[dict]:
        if not pmcids: return []
        print(f"Fetching best available text for {len(pmcids)} PMC articles...")
        articles = []
        try:
            fetch_params = {"db": "pmc", "id": ",".join(pmcids), "retmode": "xml", "tool": self.tool_name, "email": self.email}
            fetch_response = requests.get(self.ncbi_base_url + "efetch.fcgi", params=fetch_params)
            fetch_response.raise_for_status()
            root = ET.fromstring(fetch_response.content)

            for article_xml in root.findall('.//article'):
                pmid_node = article_xml.find(".//*[@pub-id-type='pmid']")
                title_node = article_xml.find('.//article-title')
                
                # Use our new, robust parser
                content = self._parse_best_available_content(article_xml)

                articles.append({
                    "id": pmid_node.text if pmid_node is not None else "N/A",
                    "title": "".join(title_node.itertext()).strip() if title_node is not None else "No Title",
                    "content": content,
                    "source_type": "PMC" # Source is PMC, content could be full-text or abstract
                })
        except (requests.exceptions.RequestException, ET.ParseError) as e:
            print(f"Error fetching/parsing from PMC: {e}")
        
        return articles

    def _fetch_abstracts_from_pubmed(self, pmids: List[str]) -> List[dict]:
        if not pmids: return []
        print(f"Fetching abstracts for {len(pmids)} PubMed articles...")
        articles = []
        try:
            fetch_params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml", "tool": self.tool_name, "email": self.email}
            fetch_response = requests.get(self.ncbi_base_url + "efetch.fcgi", params=fetch_params)
            fetch_response.raise_for_status()
            root = ET.fromstring(fetch_response.content)
            for article_xml in root.findall('.//PubmedArticle'):
                pmid_node = article_xml.find('.//PMID')
                title_node = article_xml.find('.//ArticleTitle')

                # Use our new, robust parser here as well for consistency
                content = self._parse_best_available_content(article_xml)

                articles.append({
                    "id": pmid_node.text if pmid_node is not None else "N/A",
                    "title": "".join(title_node.itertext()).strip() if title_node is not None else "No Title",
                    "content": content,
                    "source_type": "PubMed (Abstract)"
                })
        except (requests.exceptions.RequestException, ET.ParseError) as e:
            print(f"Error fetching/parsing from PubMed: {e}")
        return articles
            
    def search_and_rank_passages(self, query: str, user_question: str, max_articles: int = 5, max_passages_per_llm_context: int = 5, rank: bool = True) -> List[dict]:
        """
        The main pipeline: Search for docs, fetch, chunk, and re-rank passages.

        Args:
            query (str): The search query (can be natural language or Boolean).
            max_articles (int): Number of top articles to fetch from the initial search.
            max_passages_per_llm_context (int): Final number of top passages to return for the LLM.

        Returns:
            A list of the most relevant passage dictionaries.
        """
        print(f"\n--- Stage 1: Broad Search on Europe PMC for query: '{query}' ---")
        params = {
            "query": query,
            "resultType": "core",  # 'core' returns full metadata including abstracts
            "format": "json",
            "pageSize": max_articles
        }
        
        final_articles = []
        pmids_to_process = []
        
        try:
            response = requests.get(self.europe_pmc_url, params=params)
            response.raise_for_status()
            data = response.json()
            results = data.get("resultList", {}).get("result", [])

            if not results:
                print("No results found on Europe PMC.")
                return []

            # --- Triage Results: Separate Pre-prints and standard articles ---
            for result in results:
                if result.get("source") == "PPR":
                    # It's a pre-print, process it directly
                    abstract = result.get("abstractText", "No abstract available.")
                    abstract = re.sub('<[^<]+?>', '', abstract) # Clean HTML tags
                    final_articles.append({
                        "id": result.get("id"),
                        "title": result.get("title", "No Title"),
                        "content": abstract,
                        "source_type": "Pre-print"
                    })
                elif "pmid" in result:
                    # It's a standard article, queue its PMID for batch processing
                    pmids_to_process.append(result["pmid"])
            
            # --- Batch Process Standard Articles for Full Text/Abstracts ---
            if pmids_to_process:
                print(f"Found {len(pmids_to_process)} standard articles to process further.")
                time.sleep(0.4)
                pmid_to_pmcid_map = self._convert_pmids_to_pmcids(pmids_to_process)
                pmcids_to_fetch = list(pmid_to_pmcid_map.values())
                
                # Fetch full text for articles that have a PMCID
                full_text_articles = self._fetch_full_text_from_pmc(pmcids_to_fetch)
                final_articles.extend(full_text_articles)
                
                # Determine which PMIDs still need an abstract
                fetched_pmids = {article['id'] for article in full_text_articles}
                pmids_for_fallback = [pmid for pmid in pmids_to_process if pmid not in fetched_pmids]
                
                # Fetch abstracts for the rest
                if pmids_for_fallback:
                    time.sleep(0.4)
                    abstract_articles = self._fetch_abstracts_from_pubmed(pmids_for_fallback)
                    final_articles.extend(abstract_articles)

        except requests.exceptions.RequestException as e:
            print(f"Error during Europe PMC search: {e}")
            return []

        if not final_articles:
            print("No usable content retrieved from any source.")
            return []
        if rank:

            print(f"\n--- Stage 2: Chunking & Re-ranking {len(final_articles)} Documents ---")

            # Step 2a: Chunk all documents into passages
            all_passages = []
            for doc in final_articles:
                chunks = self._chunk_text(doc['content'])
                for chunk in chunks:
                    all_passages.append({
                        "text": chunk,
                        "article_id": doc.get('id', 'N/A'),
                        "article_title": doc.get('title', 'No Title'),
                        "source_type": doc.get('source_type', 'Unknown')
                    })
            
            if not all_passages: return []
            print(f"Created {len(all_passages)} passages for re-ranking.")
            
            # Step 2b: Embed all passages and the query
            print("Embedding passages and query (this may take a moment)...")
            passage_texts = [p['text'] for p in all_passages]
            passage_embeddings = np.array(self.embedding_model.embed_documents(passage_texts))
            query_embedding = np.array(self.embedding_model.embed_query(user_question)).reshape(1, -1) # embed user question instead of query
            
            # Step 2c: Normalize and calculate similarity
            passage_embeddings_norm = normalize(passage_embeddings, axis=1, norm='l2')
            query_embedding_norm = normalize(query_embedding, axis=1, norm='l2')
            similarities = np.dot(query_embedding_norm, passage_embeddings_norm.T)[0]
            
            # Step 2d: Get the top N most relevant passages
            top_passage_indices = np.argsort(similarities)[-max_passages_per_llm_context:][::-1]

            top_relevant_passages = [all_passages[idx] for idx in top_passage_indices]
            print(f"Selected {len(top_relevant_passages)} top passages for the LLM context.")
            return top_relevant_passages
        else:
            return final_articles
    
    def rank_passages(self, all_passages: List[str], user_question: str, max_passages_per_llm_context: int = 5) -> List[str]:
        # Step 2b: Embed all passages and the query
        print("Embedding passages and query (this may take a moment)...")
        passage_embeddings = np.array(self.embedding_model.embed_documents(all_passages))
        query_embedding = np.array(self.embedding_model.embed_query(user_question)).reshape(1, -1) # embed user question instead of query
        
        # Step 2c: Normalize and calculate similarity
        passage_embeddings_norm = normalize(passage_embeddings, axis=1, norm='l2')
        query_embedding_norm = normalize(query_embedding, axis=1, norm='l2')
        similarities = np.dot(query_embedding_norm, passage_embeddings_norm.T)[0]
        
        # Step 2d: Get the top N most relevant passages
        top_passage_indices = np.argsort(similarities)[-max_passages_per_llm_context:][::-1]

        top_relevant_passages = [all_passages[idx] for idx in top_passage_indices]
        print(f"Selected {len(top_relevant_passages)} top passages for the LLM context.")
        return top_relevant_passages
    
    def format_context_for_llm(self, articles: List[dict], rank=True) -> str:
        if not articles:
            return "No relevant information was found."
        
        if rank:
            context_str = "Here are the most relevant sections retrieved from scientific sources:\n\n"
            for i, passage in enumerate(articles, 1):
                source_type = passage['source_type']
                context_str += f"[Passage {i} from Source: {source_type}, Article ID: {passage['article_id']}]\n"
                if source_type == "Pre-print":
                    context_str += "CRITICAL WARNING: This source has NOT been peer-reviewed.\n"
                context_str += f"Original Article Title: {passage['article_title']}\n"
                context_str += f"Content: {passage['text']}\n"
                context_str += "---\n"
            return context_str
        else: 
            context_str = "Here is some relevant information retrieved from various scientific sources:\n\n"
            
            for i, article in enumerate(articles, 1):
                source_type = article['source_type']
                context_str += f"[Source {i}: {source_type}, ID: {article.get('id', 'N/A')}]\n"
                
                if source_type == "Pre-print":
                    context_str += "CRITICAL WARNING: This source has NOT been peer-reviewed.\n"

                context_str += f"Title: {article.get('title', 'No Title')}\n"
                content_snippet = (article['content'][:1500] + '...') if len(article['content']) > 1500 else article['content']
                context_str += f"Content: {content_snippet}\n"
                context_str += "---\n"
            return context_str