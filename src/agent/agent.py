import pandas as pd
import requests
from bs4 import BeautifulSoup
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import time
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

class FAUWebsiteAgent:
    def __init__(self, excel_file: str, model_name: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
        """
        Initialize the AI agent with sentence transformers
        """
        self.excel_file = excel_file
        self.df = None
        self.model = None
        self.embeddings = None
        self.load_model(model_name)  # Load model FIRST
        self.load_data()              # Then load data (needs model)
    
    def load_model(self, model_name: str):
        """Load the sentence transformer model"""
        try:
            print(f"🔄 Loading model: {model_name}")
            self.model = SentenceTransformer(model_name)
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            # Fallback to a smaller model if the specified one fails
            try:
                print("🔄 Trying fallback model: all-MiniLM-L6-v2")
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                print("✅ Fallback model loaded successfully")
            except Exception as e2:
                print(f"❌ Fallback model also failed: {e2}")
    
    def load_data(self):
        """Load and prepare the Excel data"""
        try:
            if self.model is None:
                raise ValueError("Model must be loaded before loading data")
            
            self.df = pd.read_excel(self.excel_file)
            print(f"✅ Loaded {len(self.df)} URLs from {self.excel_file}")
            
            # Clean the data
            self.df = self.df.fillna('')
            
            # Combine title and description for better search (2nd and 3rd columns)
            self.df['combined_text'] = self.df.iloc[:, 1].astype(str) + ". " + self.df.iloc[:, 2].astype(str)
            
            # Precompute embeddings for all documents
            print("🔄 Computing embeddings for all pages...")
            self.embeddings = self.model.encode(
                self.df['combined_text'].tolist(), 
                show_progress_bar=True,
                convert_to_numpy=True  # Use numpy instead of tensor
            )
            print("✅ Embeddings computed successfully")
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
    
    def find_closest_matches(self, query: str, top_k: int = 5, similarity_threshold: float = 0.3) -> pd.DataFrame:
        """
        Find the top-k most relevant pages using semantic similarity
        """
        try:
            if self.model is None or self.embeddings is None:
                raise ValueError("Model or embeddings not loaded properly")
            
            # Encode the query
            query_embedding = self.model.encode([query], convert_to_numpy=True)
            
            # Calculate cosine similarity (embeddings are already numpy)
            similarities = cosine_similarity(
                query_embedding, 
                self.embeddings
            )[0]
            
            # Get top-k indices
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            # Create results dataframe
            results = []
            for idx in top_indices:
                if similarities[idx] >= similarity_threshold:  # Only include relevant results
                    results.append({
                        'URL': self.df.iloc[idx, 0],  # 1st column
                        'Title': self.df.iloc[idx, 1],  # 2nd column
                        'Description': self.df.iloc[idx, 2],  # 3rd column
                        'Similarity_Score': float(similarities[idx]),
                        'Original_Index': idx
                    })
            
            return pd.DataFrame(results)
        
        except Exception as e:
            print(f"❌ Error finding matches: {e}")
            return pd.DataFrame()
    
    def extract_page_content(self, url: str) -> Dict[str, str]:
        """
        Extract comprehensive content from a webpage
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, timeout=15, headers=headers)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style", "nav", "footer", "header"]):
                    script.decompose()
                
                # Extract main content
                title = soup.find('title')
                title_text = title.get_text().strip() if title else ""
                
                # Try to find main content areas
                main_content = self.extract_meaningful_content(soup)
                
                return {
                    'url': url,
                    'title': title_text,
                    'content': main_content,
                    'status': 'success',
                    'content_length': len(main_content)
                }
            
            else:
                return {
                    'url': url,
                    'title': '',
                    'content': '',
                    'status': f'HTTP Error {response.status_code}',
                    'content_length': 0
                }
        
        except Exception as e:
            return {
                'url': url,
                'title': '',
                'content': '',
                'status': f'Error: {str(e)}',
                'content_length': 0
            }
    
    def extract_meaningful_content(self, soup) -> str:
        """Extract meaningful content using multiple strategies"""
        content_selectors = [
            'main', '.main-content', '#main-content', '.content',
            '.page-content', '#content', 'article', '.article',
            '.post-content', '.entry-content', '[role="main"]'
        ]
        
        for selector in content_selectors:
            elements = soup.select(selector)
            if elements:
                # Get the largest content block
                elements.sort(key=lambda x: len(x.get_text()), reverse=True)
                main_text = elements[0].get_text().strip()
                if len(main_text) > 200:  # Ensure it has substantial content
                    return self.clean_text(main_text)
        
        # Fallback: use body but remove navigation and other noise
        body = soup.find('body')
        if body:
            # Remove common noisy elements
            for noise in body.select('.navigation, .nav, .menu, .sidebar, .ads, .cookie, .modal'):
                noise.decompose()
            body_text = body.get_text().strip()
            if len(body_text) > 100:
                return self.clean_text(body_text)
        
        return ""
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-@]', '', text)
        return text.strip()
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks using sentence boundaries"""
        # Split into sentences first for better chunking
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def perform_semantic_rag(self, query: str, top_pages_df: pd.DataFrame, top_chunks: int = 5) -> Dict:
        """
        Perform semantic RAG using the same sentence transformer model
        """
        print(f"🔍 Performing semantic RAG on {len(top_pages_df)} pages...")
        
        # Extract content from all pages
        page_contents = []
        successful_urls = []
        
        for _, row in top_pages_df.iterrows():
            print(f"📄 Extracting content from: {row['URL']} (score: {row['Similarity_Score']:.3f})")
            content = self.extract_page_content(row['URL'])
            page_contents.append(content)
            
            if content['status'] == 'success' and content['content']:
                successful_urls.append(row['URL'])
            
            time.sleep(0.5)  # Be respectful
            
            # Stop if we have enough good content
            if len(successful_urls) >= 3:
                break
        
        # Combine all successful content
        all_chunks = []
        for content in page_contents:
            if content['status'] == 'success' and content['content']:
                chunks = self.chunk_text(content['content'])
                all_chunks.extend([(chunk, content['url'], content['title']) for chunk in chunks])
        
        if not all_chunks:
            return {
                'success': False,
                'error': 'No content could be extracted from the pages',
                'pages_processed': page_contents
            }
        
        # Use sentence transformers to find most relevant chunks
        chunk_texts = [chunk[0] for chunk in all_chunks]
        chunk_embeddings = self.model.encode(chunk_texts, convert_to_tensor=True)
        query_embedding = self.model.encode([query], convert_to_tensor=True)
        
        # Calculate similarities
        chunk_similarities = cosine_similarity(
            query_embedding.cpu().numpy(), 
            chunk_embeddings.cpu().numpy()
        )[0]
        
        # Get top chunks
        top_chunk_indices = np.argsort(chunk_similarities)[-top_chunks:][::-1]
        
        # Prepare context from top chunks
        context_chunks = []
        for idx in top_chunk_indices:
            if chunk_similarities[idx] > 0.3:  # Similarity threshold
                chunk_text, url, title = all_chunks[idx]
                context_chunks.append({
                    'text': chunk_text,
                    'similarity': float(chunk_similarities[idx]),
                    'source': url,
                    'title': title
                })
        
        # Generate answer
        answer = self.generate_semantic_answer(query, context_chunks)
        
        return {
            'success': True,
            'query': query,
            'answer': answer,
            'context_chunks': context_chunks,
            'top_matches': top_pages_df.to_dict('records'),
            'pages_processed': page_contents
        }
    
    def generate_semantic_answer(self, query: str, context_chunks: List[Dict]) -> str:
        """
        Generate an answer based on the most relevant context chunks
        """
        if not context_chunks:
            return "I couldn't find enough relevant information to answer your question."
        
        # Sort chunks by similarity
        context_chunks.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Build the answer
        answer_parts = []
        answer_parts.append(f"**Question:** {query}\n")
        answer_parts.append("**Answer based on FAU website information:**\n")
        
        # Use top 3 chunks for answer
        used_sources = set()
        
        for i, chunk in enumerate(context_chunks[:3]):
            if chunk['source'] not in used_sources:
                answer_parts.append(f"{chunk['text']}")
                used_sources.add(chunk['source'])
        
        # Add sources
        answer_parts.append("\n**Sources:**")
        for chunk in context_chunks[:3]:
            if chunk['source']:
                answer_parts.append(f"- {chunk['source']}")
        
        answer_parts.append(f"\n*This information was extracted from {len(used_sources)} relevant FAU web pages using semantic search.*")
        
        return "\n".join(answer_parts)
    
    def query(self, user_query: str, top_k: int = 5) -> Dict:
        """
        Main method to handle user queries with semantic search
        """
        print(f"🤔 Processing query: '{user_query}'")
        
        if self.model is None:
            return {
                'success': False,
                'error': 'AI model not available. Please check the model installation.',
                'query': user_query
            }
        
        # Step 1: Find closest matches using semantic search
        print("🔎 Performing semantic search...")
        closest_matches = self.find_closest_matches(user_query, top_k=top_k)
        
        if closest_matches.empty:
            return {
                'success': False,
                'error': 'No relevant pages found for your query. Try using different keywords.',
                'query': user_query
            }
        
        print(f"✅ Found {len(closest_matches)} relevant pages (semantic similarity):")
        for _, row in closest_matches.iterrows():
            print(f"   - {row['Title'][:60]}... (score: {row['Similarity_Score']:.3f})")
        
        # Step 2: Perform semantic RAG on top pages
        rag_result = self.perform_semantic_rag(user_query, closest_matches)
        
        return rag_result

# Interactive version
def interactive_agent():
    """Run the agent in interactive mode"""
    excel_file = r"/home/ubuntu/projects/FAUgen-AI/src/excel_processing/metadata_title_desc.xlsx"  # Replace with your file
    
    try:
        print("🚀 Loading FAU Website AI Agent with Sentence Transformers...")
        agent = FAUWebsiteAgent(excel_file)
        
        print("\n" + "="*60)
        print("🤖 FAU AI Agent Ready! Type your questions about FAU.")
        print("Type 'quit' or 'exit' to stop.")
        print("="*60)
        
        while True:
            user_query = input("\n🎯 Your question: ").strip()
            
            if user_query.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye!")
                break
            
            if not user_query:
                continue
            
            print("🔄 Processing...")
            start_time = time.time()
            
            result = agent.query(user_query, top_k=5)
            
            end_time = time.time()
            print(f"\n⏱️  Processing time: {end_time - start_time:.2f} seconds")
            
            if result['success']:
                print("\n" + "="*60)
                print("✅ **ANSWER:**")
                print(result['answer'])
                print("="*60)
            else:
                print(f"\n❌ Error: {result.get('error', 'Unknown error')}")
            
            print(f"\n📊 Stats: {len(result.get('context_chunks', []))} context chunks used")
    
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    # First, make sure to install the required packages:
    # uv add sentence-transformers pandas beautifulsoup4 requests numpy scikit-learn torch
    
    interactive_agent()