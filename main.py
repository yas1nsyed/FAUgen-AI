import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agent.cleaner_agent import FAURAGAgent
from llm_client import make_llm


def main():
    """Main orchestration: cleaner_agent → llm_client"""
    
    excel_file = Path(__file__).parent / "src/excel_processing/metadata_title_desc.xlsx"
    
    try:
        print("🚀 FAU AI Assistant")
        print("=" * 70)
        
        # Initialize agents
        agent = FAURAGAgent(str(excel_file))
        llm = make_llm(provider="gemini")
        
        print("✅ System ready!\n")
        
        # Interactive loop
        while True:
            question = input("❓ Ask about FAU (or 'quit'):\n> ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if not question:
                continue
            
            # Get relevant URLs from cleaner_agent
            top_df, scores = agent.top_descriptions(question, k=5)
            urls = top_df.iloc[:, 0].tolist()
            
            # Get context from cleaner_agent
            context = agent.rag(question, urls)
            
            # Pass to LLM
            prompt = f"""Answer based on this FAU context:

**Question:** {question}

**Context:**
{context}

**Answer:**"""
            
            answer = llm.invoke(prompt)
            print(f"\n{answer}\n")
            print("=" * 70)
    
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
