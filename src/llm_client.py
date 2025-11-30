from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from dotenv import load_dotenv

# Load .env from the project root directory
load_dotenv()

class AgentManager:
    def __init__(self):
        self.llm_model = None
        self.agent = None
        self.chat_history = [] 
    
    def get_llm(self, provider: str, model: str):
        if provider == "OpenAI":
            self.llm_model = ChatOpenAI(model= model, temperature=0.3, timeout=30)
        elif provider == "Google":
            self.llm_model = ChatGoogleGenerativeAI(model= model, temperature=0.3, timeout=30)
        elif provider == "Ollama":
            self.llm_model = ChatOllama(model= model, temperature=0.3, base_url="http://localhost:11434")
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def create_agent(self):
        self.agent = create_agent(model=self.llm_model, tools=[], system_prompt="You are a helpful machine that helps students to guide through their journey to discover Friedrich-Alexander-Universität (FAU) Erlangen-Nürnberg university.")
        return self.agent

    def interact_with_agent(self, message):
        if self.agent is None and self.llm_model is None:
            reply = "Agent/LLM not configured"
            self.chat_history.append({"role": "user", "content": message})
            self.chat_history.append({"role": "assistant", "content": reply})
            return self.chat_history
        # add user message
        self.chat_history.append({"role": "user", "content": message})

        try:
            messages = {"messages": [{"role": "user", "content": message}]}
            result = self.agent.invoke(messages)
            print(result)
        except Exception as e:
            return f"Error during agent interaction: {str(e)}"
        
        reply = result['messages'][-1].content

        # normalize result -> reply string
        if 'reply' not in locals():
            if isinstance(result, str):
                reply = result
            elif isinstance(result, dict):
                # try common fields
                reply = result.get("text") or result.get("content") or str(result)
            elif hasattr(result, "text"):
                reply = getattr(result, "text")
            else:
                reply = str(result)

        # append assistant reply and return history
        self.chat_history.append({"role": "assistant", "content": reply})
        return self.chat_history

agent_manager = AgentManager()

# ### Stale
# # Try to import Gemini (optional)
# try:
#     import google.generativeai as genai
#     GEMINI_AVAILABLE = True
# except ImportError:
#     GEMINI_AVAILABLE = False


# class GeminiLLM:
#     """Simple wrapper for Google's Generative AI"""
#     def __init__(self, model: str = "gemini-3-pro-preview", api_key: str = None):
#         if not GEMINI_AVAILABLE:
#             raise ImportError("google-generativeai not installed. Run: uv add google-generativeai")
        
#         # Get API key from parameter or .env file
#         self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
#         if not self.api_key:
#             raise ValueError("GOOGLE_API_KEY not set in .env file")
        
#         genai.configure(api_key=self.api_key)
#         self.model = genai.GenerativeModel(model)
    
#     def invoke(self, prompt: str) -> str:
#         """Generate response using Gemini"""
#         response = self.model.generate_content(prompt)
#         return response.text


# def make_llm(provider: str = "gemini", model: str = None, api_key: str = None):
#     """Create an LLM instance based on provider (Gemini is default)"""
#     provider = provider.lower()
    
#     if provider == "gemini":
#         model = model or "gemini-2.5-flash"
#         return GeminiLLM(model=model, api_key=api_key)
    
#     elif provider == "openai":
#         model = model or "gpt-4"
#         api_key = api_key or os.getenv("OPENAI_API_KEY")
#         return ChatOpenAI(model_name=model, api_key=api_key)
    
#     elif provider == "ollama":
#         model = model or "llama2"
#         return ChatOllama(model=model, base_url="http://localhost:11434")
    
#     else:
#         raise ValueError(f"❌ Unknown provider: {provider}")


# def create_fau_agent(llm, system_prompt: str = "You are a helpful FAU AI Assistant."):
#     """Create a simple agent wrapper"""
#     return {
#         "llm": llm,
#         "system_prompt": system_prompt,
#         "invoke": lambda query: llm.invoke(query)
#     }


# # Example usage
# if __name__ == "__main__":
#     # Gemini is now the default (loads API key from .env)
#     try:
#         print(f"🔄 Loading .env from: {Path(__file__).parent.parent / '.env'}")
#         llm = make_llm()  # Uses Gemini by default
        
#         # Create agent
#         agent = create_fau_agent(
#             llm=llm,
#             system_prompt="You are a helpful assistant for FAU students."
#         )
        
#         # Use the agent
#         result = agent["invoke"]("Hello, what can you help me with?")
#         print(result)
    
#     except (ImportError, ValueError) as e:
#         print(f"❌ Error: {e}")
#         print("Falling back to OpenAI...")
        
#         try:
#             llm = make_llm("openai")
#             agent = create_fau_agent(llm)
#             result = agent["invoke"]("Hello, what can you help me with?")
#             print(result)
#         except Exception as e2:
#             print(f"❌ OpenAI also failed: {e2}")
