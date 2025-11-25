# embeddings.py
import faiss
import pandas as pd
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer

EMBEDDING_DIM = 768
INDEX_PATH = "data/embeddings.index"
METADATA_PATH = "data/metadata.parquet"

class DocumentStore:
    def save(self, df: pd.DataFrame):
        Path(METADATA_PATH).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(METADATA_PATH, index=False)
        print(f"[DocumentStore] Saved metadata → {METADATA_PATH}")

class EmbeddingStore:
    def build_and_save(self, embeddings: torch.Tensor):
        Path(INDEX_PATH).parent.mkdir(parents=True, exist_ok=True)
        vecs = embeddings.cpu().numpy().astype("float32")
        faiss.normalize_L2(vecs)
        dim = vecs.shape[1]
        index = faiss.IndexHNSWFlat(dim, 32)
        index.hnsw.efConstruction = 400
        index.add(vecs)
        faiss.write_index(index, INDEX_PATH)
        print(f"[EmbeddingStore] Saved FAISS index → {INDEX_PATH}")

def build_embeddings_from_excel(excel_file: str):
    """Build embeddings from Excel and save them to disk."""
    print(f"[embeddings.py] Loading Excel → {excel_file}")
    df = pd.read_excel(excel_file).fillna("")
    corpus = (df.iloc[:, 1].astype(str) + " " + df.iloc[:, 2].astype(str)).tolist()
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    embeddings = model.encode(corpus, convert_to_tensor=True, batch_size=64, show_progress_bar=True)

    # Save
    doc_store = DocumentStore()
    doc_store.save(df)
    emb_store = EmbeddingStore()
    emb_store.build_and_save(embeddings)


# build db runner - Uncomment the 2 lines and run embeddings.py to make new embeddings
# from embeddings import build_embeddings_from_excel
# build_embeddings_from_excel("src/excel_processing/aces_metadata.xlsx")
