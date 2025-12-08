# rag_agent.py
import faiss
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import requests
from bs4 import BeautifulSoup
from .embeddings import DocumentStore, EmbeddingStore
from src.tools.scraper import WebsiteScraper
from huggingface_hub import hf_hub_download
import fitz
import requests

# Download the index and metadata from hf
INDEX_PATH = hf_hub_download(
    repo_id="Yas1n/RAG_AUgen-AI",
    filename="data/embeddings.index",
)
METADATA_PATH = hf_hub_download(
    repo_id="Yas1n/RAG_AUgen-AI",
    filename="data/metadata.parquet",
)

# INDEX_PATH = "/home/ubuntu/projects/FAUgen-AI/data/embeddings.index"
# METADATA_PATH = "/home/ubuntu/projects/FAUgen-AI/data/metadata.parquet"

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
        return (self.df.iloc[indices, 3].astype(str)).tolist()
    
    def pdf_start_page(self, indices):
        return (self.df.iloc[indices, 4].astype(int)).tolist()

    def pdf_end_page(self, indices):
        return (self.df.iloc[indices, 5].astype(int)).tolist()

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

    def search(self, query_vec: np.ndarray, top_k: int = 5, efSearch: int = 400):
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
        # self.model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
        self.model = SentenceTransformer("Qwen/Qwen3-Embedding-4B")

    def top_k_unique(self, links, texts, scores, k):
        seen = set()
        unique_results = []

        for link, text, score in zip(links, texts, scores):
            if link not in seen:
                seen.add(link)
                unique_results.append((link, text, float(score)))
            if len(unique_results) == k:
                break

        return unique_results


    def query(self, query_text: str, top_k: int, return_indices=False):
        search_k = 100

        q_emb = self.model.encode(query_text, convert_to_numpy=True).reshape(1, -1).astype("float32")
        scores, indices = self.emb_store.search(q_emb, top_k=search_k)

        links = self.doc_store.get_links(indices)
        texts = self.doc_store.get_texts(indices)

        if return_indices:
            return links, texts, scores, indices

        return links, texts, scores

        # top_unique = self.top_k_unique(links, texts, scores, k=top_k)

        # if len(top_unique) == 0:
        #     raise ValueError("No unique RAG results found.")

        # unique_links, unique_texts, unique_scores = zip(*top_unique)

        # if return_indices:
        #     return unique_links, unique_texts, unique_scores, indices

        # return unique_links, unique_texts, unique_scores

    
    def extract_pdf_pages(self, url: str, start_page: int, end_page: int) -> str:
        """
        Download PDF and extract only the pages belonging to the embedding chunk.
        """
        resp = requests.get(url)
        resp.raise_for_status()

        doc = fitz.open(stream=resp.content, filetype="pdf")

        text = ""
        for p in range(start_page, end_page + 1):
            if 0 <= p < doc.page_count:
                text += doc.load_page(p).get_text()

        return text.strip()

    def scrape(self, url: str) -> str:                                
        try:
            text = WebsiteScraper().scrape_website(url)
            return text
        except Exception as e:
            return f"[Error fetching URL: {e}]"

    def rag(self, query_text: str, top_k: int):
        """
        Retrieve links via FAISS.
        If link is a PDF → load PDF → extract only the pages belonging to the embedding chunk.
        Otherwise → scrape website normally.
        """

        # FIX 1 — query() must return indices too
        links, texts, scores, indices = self.query(query_text, top_k=top_k, return_indices=True)

        # FIX 2 — use SAME filtered indices for page lookup
        start_pages = self.doc_store.pdf_start_page(indices)
        end_pages = self.doc_store.pdf_end_page(indices)

        rag_outputs = []

        for i, link in enumerate(links):

            # Case 1: PDF links
            if link.lower().endswith(".pdf"):
                try:
                    # extract the PDF text
                    extracted_text = self.extract_pdf_pages(
                        url=link,
                        start_page=start_pages[i],
                        end_page=end_pages[i]
                    )

                    rag_outputs.append(
                        f"[RAG output] Link: {link}\n{extracted_text}"
                    )

                except Exception as e:
                    rag_outputs.append(
                        f"[RAG output] Link: {link}\nPDF read error: {e}"
                    )

            # Case 2: Website links
            else:
                try:
                    website_text = self.scrape(link)
                    rag_outputs.append(
                        f"[RAG output] Link: {link}\n{website_text}"
                    )
                except Exception as e:
                    rag_outputs.append(
                        f"[RAG output] Link: {link}\nWebsite scrape error: {e}"
                    )

        return rag_outputs

# import pandas as pd
# from pathlib import Path

# METADATA_PATH = "/home/ubuntu/projects/FAUgen-AI/data/metadata.parquet"

# # Initialize and load
# store = DocumentStore()
# store.load()

# n = len(store.df)
# # Suppose you want to check the first 5 rows
# indices = list(range(n-5,n))

# links = store.get_links(indices)
# texts = store.get_texts(indices)
# start_pages = store.pdf_start_page(indices)
# end_pages = store.pdf_end_page(indices)

# print("Links:", links)
# print("Texts:", texts)
# print("Start pages:", start_pages)
# print("End pages:", end_pages)