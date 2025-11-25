# rag_agent.py
import faiss
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import requests
from bs4 import BeautifulSoup
from .embeddings import DocumentStore, EmbeddingStore


INDEX_PATH = "data/embeddings.index"
METADATA_PATH = "data/metadata.parquet"

class DocumentStore:
    def __init__(self):
        self.df = None

    def load(self):
        if Path(METADATA_PATH).exists():
            self.df = pd.read_parquet(METADATA_PATH)
            print(f"[DocumentStore] Loaded metadata → {METADATA_PATH}")
        else:
            raise FileNotFoundError(f"{METADATA_PATH} not found. Run embeddings builder first.")

    def get_links(self, indices):
        return self.df.iloc[indices, 0].tolist()

    def get_texts(self, indices):
        return (self.df.iloc[indices, 1].astype(str) + " " +
                self.df.iloc[indices, 2].astype(str)).tolist()

class EmbeddingStore:
    def __init__(self):
        self.index = None

    def load(self):
        if Path(INDEX_PATH).exists():
            print(f"[EmbeddingStore] Loading FAISS index → {INDEX_PATH}")
            self.index = faiss.read_index(INDEX_PATH)
            print("[EmbeddingStore] Index loaded.")
        else:
            raise FileNotFoundError(f"{INDEX_PATH} not found. Run embeddings builder first.")

    def search(self, query_vec: np.ndarray, top_k: int = 5, efSearch: int = 200):
        faiss.normalize_L2(query_vec)
        self.index.hnsw.efSearch = efSearch
        scores, indices = self.index.search(query_vec, top_k)
        return scores[0], indices[0]

class RAGAgent:
    def __init__(self):
        self.doc_store = DocumentStore()
        self.doc_store.load()
        self.emb_store = EmbeddingStore()
        self.emb_store.load()
        self.model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

    def query(self, query_text: str, top_k: int = 5):
        q_emb = self.model.encode(query_text, convert_to_numpy=True).reshape(1, -1).astype("float32")
        scores, indices = self.emb_store.search(q_emb, top_k=top_k)
        links = self.doc_store.get_links(indices)
        texts = self.doc_store.get_texts(indices)
        return links, texts, scores

    def scrape(self, url: str) -> str:                                 # NEW SCRAPER COMES HERE
        try:
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            # Remove scripts/styles
            for script in soup(["script", "style"]):
                script.extract()
            text = " ".join(soup.stripped_strings)
            return text
        except Exception as e:
            return f"[Error fetching URL: {e}]"

    def rag(self, query_text: str, top_k: int = 5):
        """
        Retrieve links via FAISS + scrape each link's content for RAG.
        """
        links, texts, scores = self.query(query_text, top_k=top_k)
        rag_outputs = []
        for link in links:
            website_text = self.scrape(link)
            # Optional: summarize or send to LLM here
            snippet = website_text  
            rag_outputs.append(f"[RAG output] Link: {link}\n{snippet}...")
        return rag_outputs
