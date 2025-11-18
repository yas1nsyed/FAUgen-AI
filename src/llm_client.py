import os
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from dotenv import load_dotenv

# Load .env from the project root directory
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# Try to import Gemini (optional)
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class GeminiLLM:
    """Simple wrapper for Google's Generative AI"""
    def __init__(self, model: str = "gemini-3-pro-preview", api_key: str = None):
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai not installed. Run: uv add google-generativeai")
        
        # Get API key from parameter or .env file
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not set in .env file")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model)
    
    def invoke(self, prompt: str) -> str:
        """Generate response using Gemini"""
        response = self.model.generate_content(prompt)
        return response.text


def make_llm(provider: str = "gemini", model: str = None, api_key: str = None):
    """Create an LLM instance based on provider (Gemini is default)"""
    provider = provider.lower()
    
    if provider == "gemini":
        model = model or "gemini-2.5-flash"
        return GeminiLLM(model=model, api_key=api_key)
    
    elif provider == "openai":
        model = model or "gpt-4"
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        return ChatOpenAI(model_name=model, api_key=api_key)
    
    elif provider == "ollama":
        model = model or "llama2"
        return ChatOllama(model=model, base_url="http://localhost:11434")
    
    else:
        raise ValueError(f"❌ Unknown provider: {provider}")


def create_fau_agent(llm, system_prompt: str = "You are a helpful FAU AI Assistant."):
    """Create a simple agent wrapper"""
    return {
        "llm": llm,
        "system_prompt": system_prompt,
        "invoke": lambda query: llm.invoke(query)
    }


# Example usage
if __name__ == "__main__":
    # Gemini is now the default (loads API key from .env)
    try:
        print(f"🔄 Loading .env from: {Path(__file__).parent.parent / '.env'}")
        llm = make_llm()  # Uses Gemini by default
        
        # Create agent
        agent = create_fau_agent(
            llm=llm,
            system_prompt="You are a helpful assistant for FAU students."
        )
        
        # Use the agent
        result = agent["invoke"]("Hello, what can you help me with?")
        print(result)
    
    except (ImportError, ValueError) as e:
        print(f"❌ Error: {e}")
        print("Falling back to OpenAI...")
        
        try:
            llm = make_llm("openai")
            agent = create_fau_agent(llm)
            result = agent["invoke"]("Hello, what can you help me with?")
            print(result)
        except Exception as e2:
            print(f"❌ OpenAI also failed: {e2}")
