# run_rag.py
from rag_agent import RAGAgent

def main():
    print("=== RAG Agent Interactive Runner ===")
    agent = RAGAgent()  # Load FAISS + metadata

    while True:
        query = input("\nEnter your query (or 'exit' to quit): ").strip()
        if query.lower() in ["exit", "quit"]:
            print("Exiting RAG agent...")
            break

        print("\n[INFO] Searching database and fetching website content...")
        try:
            results = agent.rag(query, top_k=5)
            print("\nTop results:\n")
            for i, r in enumerate(results, 1):
                print(f"Result {i}:\n{r}\n{'-'*80}")
        except Exception as e:
            print(f"[ERROR] {e}")

if __name__ == "__main__":
    main()
