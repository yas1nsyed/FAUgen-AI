from pathlib import Path
from src.agent.rag_agent import RAGAgent
from src.llm_client import make_llm, create_fau_agent

def main():
    print("=== FAU RAG + LLM Pipeline ===")

    # Select LLM provider
    print("Select LLM provider:")
    print("1: Gemini (default)")
    print("2: OpenAI")
    print("3: Ollama")

    choice = input("Enter choice (1/2/3): ").strip()
    if choice == "2":
        provider = "openai"
    elif choice == "3":
        provider = "ollama"
    else:
        provider = "gemini"

    model = input(f"Enter model for {provider} (leave empty for default): ").strip() or None

    try:
        llm = make_llm(provider=provider, model=model)
    except Exception as e:
        print(f"[ERROR] Failed to initialize {provider}: {e}")
        return

    agent_llm = create_fau_agent(llm, system_prompt="You are a helpful FAU AI assistant.")

    # Initialize RAG agent
    rag_agent = RAGAgent()

    print("\n=== Interactive RAG Query ===")
    print("Type your query (or 'exit' to quit):")

    while True:
        query = input("\n> ").strip()
        if query.lower() in ["exit", "quit"]:
            print("Exiting...")
            break

        print("[INFO] Performing RAG retrieval...")
        try:
            # Step 1: Get retrieved links + snippets from RAG
            rag_outputs = rag_agent.rag(query, top_k=5)

            # DEBUG: Show retrieved links
            print("\n=== Retrieved Links from VectorStore ===")
            for i, output in enumerate(rag_outputs, start=1):
                # Assuming rag_outputs are formatted like: "[RAG output] Link: <url>\n<snippet>..."
                link_line = output.split("\n")[0]
                url = link_line.replace("[RAG output] Link: ", "")
                print(f"{i}. {url}")
            print("========================================\n")

            # Construct LLM prompt
            context = "\n\n".join(rag_outputs)
            prompt = (
                f"You are a helpful assistant. Answer the user query based on the context below.\n\n"
                f"Context:\n{context}\n\n"
                f"User query: {query}\n\nAnswer:"
            )

            # Call LLM
            answer = agent_llm["invoke"](prompt)
            print("\n=== LLM Answer ===")
            print(answer)

        except Exception as e:
            print(f"[ERROR] RAG+LLM failed: {e}")

if __name__ == "__main__":
    main()
