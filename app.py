import sys
from pathlib import Path
import gradio as gr

# Add src to path
sys.path.insert(0, str(Path(_file_).parent / "src"))

from agent.cleaner_agent import FAURAGAgent
from llm_client import make_llm


# Initialize agents once when the app starts
excel_file = Path(_file_).parent / "src/excel_processing/metadata_title_desc.xlsx"

try:
    print("🔄 Loading FAU RAG Agent...")
    agent = FAURAGAgent(str(excel_file))
    print("✅ RAG Agent loaded!")
except Exception as e:
    print(f"❌ Error loading RAG Agent: {e}")
    agent = None

try:
    print("🔄 Loading LLM (Gemini)...")
    llm = make_llm(provider="gemini")
    print("✅ Gemini loaded!")
except Exception as e:
    print(f"⚠  Gemini failed, switching to OpenAI: {e}")
    try:
        llm = make_llm(provider="openai")
        print("✅ OpenAI loaded!")
    except Exception as e2:
        print(f"❌ OpenAI also failed: {e2}")
        llm = None


def chat_fn(message, history):
    """Chat handler for Gradio - integrates RAG + LLM"""
    
    if not agent or not llm:
        return "❌ System not initialized. Check logs."
    
    if not message.strip():
        return "Please enter a question."
    
    try:
        # Step 1: Get top 5 relevant URLs from RAG
        top_df, scores = agent.top_descriptions(message, k=5)
        urls = top_df.iloc[:, 0].tolist()
        
        # Step 2: Extract content chunks
        chunks = []
        sources = []
        
        for url in urls:
            text = agent.fetch(url)
            if len(text) < 200:
                continue
            ch = agent.chunk(text, size=350)
            chunks.extend(ch)
            sources.extend([url] * len(ch))
        
        if not chunks:
            return "❌ Could not find relevant information."
        
        # Step 3: Extract top 5 relevant chunks
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        
        chunk_embs = agent.bi_encoder.encode(chunks, convert_to_numpy=True)
        q_emb = agent.bi_encoder.encode([message], convert_to_numpy=True)
        sims = cosine_similarity(q_emb, chunk_embs)[0]
        top_idx = np.argsort(sims)[-5:][::-1]
        
        relevant_chunks = [chunks[i] for i in top_idx]
        relevant_sources = list(set([sources[i] for i in top_idx]))
        
        context = "\n\n".join(relevant_chunks)
        
        # Step 4: Generate answer with LLM
        prompt = f"""You are a helpful FAU (Friedrich-Alexander-Universität Erlangen-Nürnberg) assistant.
Based on the following context from FAU websites, answer the user's question clearly and concisely.

*User Question:* {message}

*Context from FAU Websites:*
{context}

*Instructions:*
- Answer directly based on the provided context
- Be concise but informative
- Use bullet points if listing information
- Cite sources when relevant

*Answer:*"""
        
        answer = llm.invoke(prompt)
        
        # Add sources
        sources_text = "\n\n*📚 Sources:*\n" + "\n".join(f"🔗 {s}" for s in relevant_sources)
        
        return answer + sources_text
    
    except Exception as e:
        return f"❌ Error processing query: {e}"


### Gradio UI
with gr.Blocks(title="FAUgen AI") as demo:
    gr.Markdown("# 👁 FAUgen AI — FAU Search Assistant\nAsk anything about FAU websites, programs, research, campus life, etc.")
    
    with gr.Row():
        chatbot = gr.Chatbot(height=500, scale=1, type="messages")
    
    with gr.Row():
        msg = gr.Textbox(
            placeholder="Ask something about FAU...",
            label="Your Question",
            scale=4
        )
        submit_btn = gr.Button("Send", scale=1)
    
    clear = gr.Button("Clear Chat")
    
    # Handle chat submission
    def handle_submit(user_msg, chat_history):
        if not user_msg.strip():
            return chat_history
        
        # Get bot response
        bot_response = chat_fn(user_msg, chat_history)
        
        # Append to history in OpenAI message format
        chat_history.append({"role": "user", "content": user_msg})
        chat_history.append({"role": "assistant", "content": bot_response})
        
        return chat_history
    
    # Submit on button click or Enter key
    msg.submit(handle_submit, [msg, chatbot], chatbot)
    submit_btn.click(handle_submit, [msg, chatbot], chatbot)
    
    # Clear input after submit
    msg.submit(lambda: "", None, msg)
    submit_btn.click(lambda: "", None, msg)
    
    # Clear chat history
    clear.click(lambda: [], None, chatbot)


# Launch app
if _name_ == "_main_":
    demo.launch(share=True)