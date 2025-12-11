# embeddings.py
import faiss
import pandas as pd
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer

EMBEDDING_DIM = 1024
INDEX_PATH = "data/embeddings.index"
METADATA_PATH = "data/metadata.parquet"

class DocumentStoreSave:
    def save(self, df: pd.DataFrame):
        Path(METADATA_PATH).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(METADATA_PATH, index=False)
        print(f"[DocumentStore] Saved metadata → {METADATA_PATH}")

class EmbeddingStoreSave:
    
    def build_and_save(self, embeddings: torch.Tensor):
        Path(INDEX_PATH).parent.mkdir(parents=True, exist_ok=True)

        vecs = embeddings.cpu().numpy().astype("float32")
        faiss.normalize_L2(vecs)

        dim = vecs.shape[1]
        index = faiss.IndexHNSWFlat(dim, 64)
        index.hnsw.efConstruction = 1000

        index.add(vecs)
        faiss.write_index(index, INDEX_PATH)

        print(f"[EmbeddingStore] Saved FAISS index → {INDEX_PATH}")


    @staticmethod
    def chunk_text(text, chunk_size=2000, overlap=200):
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += chunk_size - overlap

        return chunks
    
    @staticmethod
    def clean_page_value(value):
        """Convert Excel page column into a safe integer."""
        try:
            if value is None:
                return -1
            s = str(value).strip()
            if s == "":
                return -1
            return int(float(s))   # handles "10.0", 42.0 values
        except Exception:
            return -1

    @staticmethod
    def build_embeddings_from_excel(excel_file: str):
        print(f"[embeddings.py] Loading Excel → {excel_file}")
        df = pd.read_excel(excel_file).fillna("")

        # model = SentenceTransformer(
        #     "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        # )
        model = SentenceTransformer(
            "Qwen/Qwen3-Embedding-4B"
        )

        all_chunks = []
        metadata = []

        for idx, row in df.iterrows():
            doc_id = str(row[0])      # Link or ID in column 1
            full_text = str(row[5])   # Main text in column 6
            pdf_page_start = EmbeddingStoreSave.clean_page_value(row[6])
            pdf_page_end = EmbeddingStoreSave.clean_page_value(row[7])

            chunks = EmbeddingStoreSave.chunk_text(full_text)

            for chunk_id, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                metadata.append({
                    "doc_id": doc_id,
                    "row": idx,
                    "chunk_id": chunk_id,
                    "chunk_text": chunk,
                    "pdf_page_start": pdf_page_start,
                    "pdf_page_end": pdf_page_end
                })

        print(f"Total chunks created: {len(all_chunks)}")

        embeddings = model.encode(
            all_chunks,
            convert_to_tensor=True,
            batch_size=4,
            show_progress_bar=True,
            normalize_embeddings=True 
        )

        # SAVE METADATA (convert list → DataFrame)
        metadata_df = pd.DataFrame(metadata)
        doc_store = DocumentStoreSave()
        doc_store.save(metadata_df)

        # SAVE EMBEDDINGS
        emb_store = EmbeddingStoreSave()
        emb_store.build_and_save(embeddings)

        print("Embedding database built successfully!")


# --- Runner ---
if __name__ == "__main__":
    EmbeddingStoreSave.build_embeddings_from_excel(
        "/home/ubuntu/projects/FAUgen-AI/src/excel_processing/combined_output.xlsx" # Enter path to the Excel file containing URLS and descriptions
    )
