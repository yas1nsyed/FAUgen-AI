import pandas as pd
import requests
from bs4 import BeautifulSoup
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from sklearn.metrics.pairwise import cosine_similarity
import re
import time
import torch


class FAURAGAgent:
    def __init__(self, excel_file):

        self.df = pd.read_excel(excel_file).fillna('')

        # -------- Stage 1: Bi-Encoder (fast retrieval) --------
        print("Loading bi-encoder model…")
        self.bi_encoder = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        )

        # -------- Stage 2: Cross-Encoder (precise ranking) --------
        print("Loading cross-encoder model…")
        self.cross_encoder = CrossEncoder(
            'cross-encoder/ms-marco-MiniLM-L6-v2'
        )

        # Corpus = title + description for best semantic signal
        self.corpus = (
            self.df.iloc[:, 1].astype(str) + " " + self.df.iloc[:, 2].astype(str)
        ).tolist()

        print("Encoding descriptions (bi-encoder)…")
        self.corpus_embs = self.bi_encoder.encode(
            self.corpus, convert_to_tensor=True, batch_size=64
        )
        print("Done.")

    # --------------------------------------------------------------
    # STEP 1 — Best bilingual search:
    # Bi-Encoder (top 50) → Cross-Encoder rerank → Top 5
    # --------------------------------------------------------------
    def top_descriptions(self, query, k=5):

        # Step 1: Similarity with bi-encoder
        q_emb = self.bi_encoder.encode(query, convert_to_tensor=True)
        sims = util.cos_sim(q_emb, self.corpus_embs)[0]

        # Take top 50 for reranking
        top_k = min(50, len(sims))
        top_ids = torch.topk(sims, k=top_k).indices.tolist()

        # Step 2: Cross-Encoder reranking
        pairs = [(query, self.corpus[i]) for i in top_ids]
        cross_scores = self.cross_encoder.predict(pairs)

        reranked = sorted(
            zip(top_ids, cross_scores), key=lambda x: x[1], reverse=True
        )

        # Final top 5 results
        final = reranked[:k]
        idxs = [x[0] for x in final]
        scores = [float(x[1]) for x in final]

        return self.df.iloc[idxs], scores

    # --------------------------------------------------------------
    # STEP 2 — Scrape webpage
    # --------------------------------------------------------------
    def fetch(self, url):
        try:
            r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla"})
            soup = BeautifulSoup(r.text, "html.parser")

            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()

            text = soup.get_text(separator=" ", strip=True)
            text = re.sub(r"\s+", " ", text)

            return text
        except:
            return ""

    # --------------------------------------------------------------
    # STEP 3 — Chunk text
    # --------------------------------------------------------------
    def chunk(self, text, size=350):
        words = text.split()
        return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]

    # --------------------------------------------------------------
    # STEP 4 — RAG over scraped pages
    # --------------------------------------------------------------
    def rag(self, query, urls):
        chunks = []
        sources = []

        for url in urls:
            print("Scraping:", url)
            text = self.fetch(url)

            if len(text) < 200:
                continue

            ch = self.chunk(text)
            chunks.extend(ch)
            sources.extend([url] * len(ch))

        if not chunks:
            return "Could not extract enough content."

        # Embed chunks
        chunk_embs = self.bi_encoder.encode(chunks, convert_to_numpy=True)
        q_emb = self.bi_encoder.encode([query], convert_to_numpy=True)

        sims = cosine_similarity(q_emb, chunk_embs)[0]
        top_idx = np.argsort(sims)[-5:][::-1]

        final_text = "\n".join([chunks[i] for i in top_idx])
        used_sources = list({sources[i] for i in top_idx})

        answer = (
            f"### Answer based on FAU Website\n\n"
            f"{final_text}\n\n"
            f"**Sources:**\n" +
            "\n".join(f"- {s}" for s in used_sources)
        )

        return answer

    # --------------------------------------------------------------
    # MAIN QUERY METHOD
    # --------------------------------------------------------------
    def query(self, question):
        print("\nFinding top 5 FAU pages…")
        top_df, scores = self.top_descriptions(question)

        urls = top_df.iloc[:, 0].tolist()
        titles = top_df.iloc[:, 1].astype(str).tolist()
        descs = top_df.iloc[:, 2].astype(str).tolist()

        print("\n========== TOP 5 MATCHES (Cross-Encoder Ranked) ==========")
        for i, (u, t, d, s) in enumerate(zip(urls, titles, descs, scores), start=1):
            print(f"\n{i}. URL: {u}")
            print(f"   Title: {t}")
            print(f"   Description: {d}")
            print(f"   Score: {s:.4f}")
        print("==========================================================\n")

        print("Running real-time RAG on these pages…")
        answer = self.rag(question, urls)
        return answer


def main():
    """Main function to run the FAU RAG Agent"""
    excel_file = "/home/ubuntu/projects/FAUgen-AI/src/excel_processing/metadata_title_desc.xlsx"
    
    try:
        print("🚀 Initializing FAU RAG Agent...")
        agent = FAURAGAgent(excel_file)
        print("✅ Agent initialized successfully!\n")
        
        # Interactive loop
        while True:
            question = input("\n🎯 Ask a question about FAU (or 'quit' to exit):\n> ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if not question:
                print("❌ Please enter a valid question.")
                continue
            
            print("\n🔄 Processing your question...")
            answer = agent.query(question)
            print("\n" + answer)
            
    except FileNotFoundError:
        print(f"❌ Excel file not found: {excel_file}")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
