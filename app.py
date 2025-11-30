import gradio as gr
from src.llm_client import agent_manager

# Model catalog for dropdowns
MODEL_CATALOG = {
    "OpenAI": ["gpt-5-mini", "gpt-4o-mini"],
    "Google": ["gemini-2.5-flash", "gemini-2.5-pro"],
    "Ollama": ["gemma3:270", "gemma3n:e4b"]
}

DEFAULT_PROVIDER = "Google"
DEFAULT_MODEL = "gemini-2.5-flash"

agent_manager.get_llm(DEFAULT_PROVIDER, model=DEFAULT_MODEL)
agent = agent_manager.create_agent()

def update_model_option(provider: str):
    """Return the value of updated model selection
    Args:
        provider (str): The LLM provider for ex: Google, OpenAI
    """
    choices = MODEL_CATALOG.get(provider, [])
    value = choices[0] if choices else "Google"
    return gr.Dropdown(choices=choices, value=value)

def set_provider_model(provider: str, model: str):
    """Set the model for the application 
    Args:
        provider (str): The LLM provider for ex: Google, OpenAI
        model (str): The LLM model ex: gemini-2.5-flash
    """
    try:
        agent_manager.get_llm(provider, model)
        agent_manager.create_agent()
        return f"LLM set to {provider} {model}"
    except Exception as e:
        return f"Failed to setup LLM Agent"

with gr.Blocks(title="AUgen AI") as demo:
    gr.Markdown("# 👁 AUgen AI — FAU Search Assistant\nAsk anything about FAU websites, programs, research, campus life, etc.")
    
    with gr.Row():
        #provider selector ui
        provider_dropdown = gr.Dropdown(label="Provider Name", choices=list(MODEL_CATALOG.keys()),value=DEFAULT_PROVIDER)
        model_dropdown = gr.Dropdown(label="Model Name", choices=MODEL_CATALOG[DEFAULT_PROVIDER], value=DEFAULT_MODEL)
        set_btn = gr.Button("Set LLM")
        status = gr.Textbox(label="LLM Status", value=f"{DEFAULT_PROVIDER}/{DEFAULT_MODEL}", interactive=False)

    provider_dropdown.change(
        fn=update_model_option,
        inputs=[provider_dropdown],
        outputs=[model_dropdown]
    )

    set_btn.click(
        fn=set_provider_model,
        inputs=[provider_dropdown, model_dropdown],
        outputs=[status]
    )


    with gr.Row():
        chatbot = gr.Chatbot(height=500, scale=1, type="messages")

    with gr.Row():
        msg = gr.Textbox(
            placeholder="Ask something about FAU...",
            label="Your Question",
            scale=4
        )
        submit_btn = gr.Button("Send", scale=1)
        submit_btn.click(
            fn=agent_manager.interact_with_agent,
            inputs=[msg],
            outputs=[chatbot]
        )


# Launch app
if __name__ == "__main__":
    demo.launch(debug=True)