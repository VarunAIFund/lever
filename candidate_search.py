#!/usr/bin/env python3
"""
Candidate Search Prototype
Uses embeddings to search candidates with natural language queries
"""

import os
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time

# Load environment variables from .env file
def load_env_file():
    """Load environment variables from .env file"""
    env_file = Path('.env')
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value

load_env_file()

# Required dependencies
try:
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("❌ Error: Required libraries not available")
    print("Install with: pip install numpy scikit-learn")
    sys.exit(1)

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("❌ Error: OpenAI library not available")
    print("Install with: pip install openai")
    sys.exit(1)

class CandidateSearchEngine:
    def __init__(self, openai_api_key: str):
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.data_dir = Path("data")
        
        # Data storage
        self.candidates = {}
        self.resumes_data = {}
        self.embeddings_data = {}
        self.search_index = []
        
        print("🔍 Initializing Candidate Search Engine...")
        self.load_all_data()
        self.build_search_index()

    def load_all_data(self):
        """Load all JSON data files"""
        print("📁 Loading data files...")
        
        # Load candidates
        candidates_file = self.data_dir / "candidates.json"
        if candidates_file.exists():
            with open(candidates_file, 'r') as f:
                candidates_list = json.load(f)
                self.candidates = {candidate['id']: candidate for candidate in candidates_list}
            print(f"✅ Loaded {len(self.candidates)} candidates")
        else:
            raise FileNotFoundError(f"Candidates file not found: {candidates_file}")
        
        # Load resume data
        resumes_file = self.data_dir / "resumes_data.json"
        if resumes_file.exists():
            with open(resumes_file, 'r') as f:
                resumes_list = json.load(f)
                self.resumes_data = {resume['candidate_id']: resume for resume in resumes_list}
            print(f"✅ Loaded resume data for {len(self.resumes_data)} candidates")
        else:
            raise FileNotFoundError(f"Resume data file not found: {resumes_file}")
        
        # Load embeddings
        embeddings_file = self.data_dir / "resume_embeddings.json"
        if embeddings_file.exists():
            with open(embeddings_file, 'r') as f:
                self.embeddings_data = json.load(f)
            print(f"✅ Loaded embeddings for {len(self.embeddings_data)} candidates")
        else:
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")

    def build_search_index(self):
        """Build search index combining all data"""
        print("🏗️  Building search index...")
        
        for candidate_id in self.candidates.keys():
            candidate = self.candidates.get(candidate_id)
            resume_data = self.resumes_data.get(candidate_id)
            embedding_data = self.embeddings_data.get(candidate_id)
            
            if not candidate or not resume_data or not embedding_data:
                continue
            
            # Extract embeddings for this candidate
            resume_embeddings = embedding_data.get('resume_embeddings', [])
            
            if not resume_embeddings:
                print(f"⚠️  No embeddings found for candidate: {candidate.get('name', 'Unknown')}")
                continue
            
            for resume_embedding in resume_embeddings:
                # Validate embedding data
                if 'embedding' not in resume_embedding:
                    print(f"⚠️  Missing embedding data for resume: {resume_embedding.get('filename', 'Unknown')}")
                    continue
                
                embedding_vector = resume_embedding['embedding']
                
                # Check if embedding is valid (not empty, not all zeros, no NaN/inf values)
                if not embedding_vector or len(embedding_vector) == 0:
                    print(f"⚠️  Empty embedding for resume: {resume_embedding.get('filename', 'Unknown')}")
                    continue
                
                # Convert to numpy array and validate
                try:
                    embedding_array = np.array(embedding_vector, dtype=np.float32)
                    
                    # Check for NaN or infinite values
                    if np.any(np.isnan(embedding_array)) or np.any(np.isinf(embedding_array)):
                        print(f"⚠️  Invalid embedding (NaN/Inf) for resume: {resume_embedding.get('filename', 'Unknown')}")
                        continue
                    
                    # Check if all values are zero
                    if np.all(embedding_array == 0):
                        print(f"⚠️  Zero embedding for resume: {resume_embedding.get('filename', 'Unknown')}")
                        continue
                    
                    # Normalize the embedding to unit vector to avoid overflow
                    embedding_norm = np.linalg.norm(embedding_array)
                    if embedding_norm > 0:
                        embedding_array = embedding_array / embedding_norm
                    else:
                        print(f"⚠️  Zero norm embedding for resume: {resume_embedding.get('filename', 'Unknown')}")
                        continue
                    
                except (ValueError, TypeError) as e:
                    print(f"⚠️  Error processing embedding for resume: {resume_embedding.get('filename', 'Unknown')} - {e}")
                    continue
                
                search_entry = {
                    'candidate_id': candidate_id,
                    'candidate_name': candidate.get('name', 'Unknown'),
                    'candidate_email': candidate.get('emails', [None])[0] if candidate.get('emails') else None,
                    'candidate_location': candidate.get('location', ''),
                    'candidate_headline': candidate.get('headline', ''),
                    'resume_id': resume_embedding['resume_id'],
                    'resume_filename': resume_embedding['filename'],
                    'text_preview': resume_embedding.get('text_preview', ''),
                    'embedding': embedding_array,
                    'embedding_dimensions': len(embedding_array),
                    'full_text_length': resume_embedding.get('full_text_length', 0)
                }
                self.search_index.append(search_entry)
        
        print(f"✅ Built search index with {len(self.search_index)} valid resume embeddings")

    def create_query_embedding(self, query: str) -> Optional[np.ndarray]:
        """Create embedding for search query"""
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=query,
                encoding_format="float"
            )
            
            embedding = np.array(response.data[0].embedding, dtype=np.float32)
            
            # Normalize the query embedding to unit vector
            embedding_norm = np.linalg.norm(embedding)
            if embedding_norm > 0:
                embedding = embedding / embedding_norm
            else:
                print(f"❌ Query embedding has zero norm")
                return None
            
            return embedding
        except Exception as e:
            print(f"❌ Error creating query embedding: {e}")
            return None

    def search(self, query: str, top_k: int = 5, min_similarity: float = 0.1, debug_mode: bool = False) -> List[Dict]:
        """Search for candidates using natural language query"""
        print(f"🔍 Searching for: '{query}' (threshold: {min_similarity:.2f})")
        
        if not self.search_index:
            print("❌ No search index available")
            return []
        
        print(f"📊 Search index contains {len(self.search_index)} embeddings")
        
        # Create query embedding
        query_embedding = self.create_query_embedding(query)
        if query_embedding is None:
            return []
        
        # Calculate similarities
        results = []
        all_similarities = []  # For debug mode
        processed_count = 0
        error_count = 0
        
        for entry in self.search_index:
            try:
                # Ensure both embeddings are valid
                candidate_embedding = entry['embedding']
                
                if candidate_embedding is None or len(candidate_embedding) == 0:
                    continue
                
                processed_count += 1
                
                # Calculate cosine similarity using dot product (since both are normalized)
                similarity = np.dot(query_embedding, candidate_embedding)
                
                # Clamp similarity to [-1, 1] range to handle floating point errors
                similarity = np.clip(similarity, -1.0, 1.0)
                
                # Store for debug output
                if debug_mode:
                    all_similarities.append({
                        'name': entry.get('candidate_name', 'Unknown'),
                        'filename': entry.get('resume_filename', 'Unknown'),
                        'similarity': float(similarity)
                    })
                
                if similarity >= min_similarity:
                    result = entry.copy()
                    result['similarity_score'] = float(similarity)
                    results.append(result)
                    
            except Exception as e:
                error_count += 1
                print(f"⚠️  Error calculating similarity for {entry.get('candidate_name', 'Unknown')}: {e}")
                continue
        
        # Debug output
        print(f"📈 Processed {processed_count} embeddings, {error_count} errors")
        
        if debug_mode and all_similarities:
            print(f"\n🔍 Debug Mode - All Similarity Scores:")
            all_similarities.sort(key=lambda x: x['similarity'], reverse=True)
            for i, sim in enumerate(all_similarities[:10]):  # Show top 10
                print(f"  {i+1}. {sim['name']} - {sim['filename']}: {sim['similarity']:.4f}")
            if len(all_similarities) > 10:
                print(f"  ... and {len(all_similarities) - 10} more")
            
            print(f"\n📊 Similarity Stats:")
            similarities_only = [s['similarity'] for s in all_similarities]
            print(f"  Max: {max(similarities_only):.4f}")
            print(f"  Min: {min(similarities_only):.4f}")
            print(f"  Avg: {np.mean(similarities_only):.4f}")
            print(f"  Above threshold ({min_similarity:.2f}): {len(results)}")
        
        print(f"✅ Found {len(results)} matches above threshold {min_similarity:.2f}")
        
        # Sort by similarity score (descending)
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return results[:top_k]

    def display_results(self, results: List[Dict], query: str):
        """Display search results in a formatted way"""
        if not results:
            print(f"\n❌ No good matches found for '{query}'")
            print("Try different keywords or lower the similarity threshold")
            return
        
        print(f"\n🎯 Found {len(results)} matches for '{query}':")
        print("=" * 80)
        
        for i, result in enumerate(results, 1):
            similarity_percent = result['similarity_score'] * 100
            
            print(f"\n{i}. {result['candidate_name']} ({similarity_percent:.1f}% match)")
            print(f"   📧 Email: {result['candidate_email'] or 'Not available'}")
            print(f"   📍 Location: {result['candidate_location'] or 'Not specified'}")
            print(f"   💼 Headline: {result['candidate_headline'] or 'Not specified'}")
            print(f"   📄 Resume: {result['resume_filename']}")
            print(f"   📝 Preview: {result['text_preview'][:200]}...")
            print(f"   🔗 Candidate ID: {result['candidate_id']}")
            print("-" * 80)

    def interactive_search(self):
        """Interactive search interface"""
        print("\n🔍 Interactive Candidate Search")
        print("=" * 50)
        print("Enter search queries to find candidates")
        print("Special commands:")
        print("  - 'quit' or 'exit' to stop")
        print("  - 'debug on/off' to toggle debug mode")
        print("  - 'threshold X' to set similarity threshold (e.g., 'threshold 0.05')")
        print("  - 'stats' to show search engine statistics")
        print("")
        print("Examples:")
        print("  - 'Python developer'")
        print("  - 'senior engineer with ML experience'")
        print("  - 'startup founder with AI background'")
        print("  - 'PhD in computer science'")
        print("")
        
        debug_mode = False
        current_threshold = 0.1
        
        print(f"🎛️  Current settings: threshold={current_threshold:.2f}, debug={debug_mode}")
        print("")
        
        while True:
            try:
                query = input("🔍 Search query: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not query:
                    continue
                
                # Handle special commands
                if query.lower().startswith('debug '):
                    mode = query.lower().split(' ', 1)[1]
                    if mode in ['on', 'true', '1']:
                        debug_mode = True
                        print("🔍 Debug mode: ON")
                    elif mode in ['off', 'false', '0']:
                        debug_mode = False
                        print("🔍 Debug mode: OFF")
                    else:
                        print("❌ Use 'debug on' or 'debug off'")
                    continue
                
                if query.lower().startswith('threshold '):
                    try:
                        new_threshold = float(query.split(' ', 1)[1])
                        if 0 <= new_threshold <= 1:
                            current_threshold = new_threshold
                            print(f"🎛️  Similarity threshold set to: {current_threshold:.2f}")
                        else:
                            print("❌ Threshold must be between 0 and 1")
                    except ValueError:
                        print("❌ Invalid threshold value. Use decimal like 0.05")
                    continue
                
                if query.lower() == 'stats':
                    self.get_search_statistics()
                    continue
                
                # Perform search
                results = self.search(query, min_similarity=current_threshold, debug_mode=debug_mode)
                self.display_results(results, query)
                
                print(f"\n{'-' * 50}")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error during search: {e}")


    def get_search_statistics(self):
        """Display search engine statistics"""
        print("\n📊 Search Engine Statistics")
        print("=" * 40)
        print(f"Total candidates: {len(self.candidates)}")
        print(f"Candidates with resumes: {len(self.resumes_data)}")
        print(f"Candidates with embeddings: {len(self.embeddings_data)}")
        print(f"Total searchable resumes: {len(self.search_index)}")
        
        if self.search_index:
            avg_dimensions = np.mean([entry['embedding_dimensions'] for entry in self.search_index])
            avg_text_length = np.mean([entry['full_text_length'] for entry in self.search_index])
            print(f"Average embedding dimensions: {avg_dimensions:.0f}")
            print(f"Average resume text length: {avg_text_length:.0f} characters")


def main():
    """Main function"""
    print("🚀 Starting Candidate Search Engine")
    print("=" * 60)
    
    # Check for OpenAI API key
    openai_api_key = os.getenv('OPEN_AI_API_KEY')
    if not openai_api_key:
        print("❌ Error: OPEN_AI_API_KEY environment variable not set")
        print("Please set it in your .env file or environment")
        return
    
    try:
        # Initialize search engine
        search_engine = CandidateSearchEngine(openai_api_key)
        
        # Display statistics
        search_engine.get_search_statistics()
        
        # Start interactive search directly
        search_engine.interactive_search()
    
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please run main.py first to extract resume data and create embeddings.")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()