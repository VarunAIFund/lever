#!/usr/bin/env python3
"""
Hybrid Candidate Search Engine
Combines semantic search (embeddings) with keyword search (TF-IDF) for better results
"""

import os
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
import re

# Load environment variables
def load_env_file():
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
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import openai
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    print(f"❌ Error: Required libraries not available: {e}")
    print("Install with: pip install numpy scikit-learn openai")
    sys.exit(1)

# ========================================
# SEARCH CONFIGURATION - MODIFY THESE VALUES
# ========================================
SEMANTIC_WEIGHT = 0.5  # How much to weight semantic similarity (0.0 to 1.0)
KEYWORD_WEIGHT = 0.5  # How much to weight keyword matching (0.0 to 1.0)
# Note: SEMANTIC_WEIGHT + KEYWORD_WEIGHT should equal 1.0

# Example configurations:
# For more semantic understanding:     SEMANTIC_WEIGHT = 0.8, KEYWORD_WEIGHT = 0.2
# For balanced approach (default):     SEMANTIC_WEIGHT = 0.6, KEYWORD_WEIGHT = 0.4
# For more exact keyword matching:     SEMANTIC_WEIGHT = 0.4, KEYWORD_WEIGHT = 0.6

class HybridCandidateSearch:
    def __init__(self, openai_api_key: str):
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.data_dir = Path("data")
        
        # Data storage
        self.candidates = {}
        self.resumes_data = {}
        self.embeddings_data = {}
        self.search_index = []
        
        # TF-IDF components
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.resume_texts = []
        
        # Search weights (semantic vs keyword) - use config values
        self.semantic_weight = SEMANTIC_WEIGHT
        self.keyword_weight = KEYWORD_WEIGHT
        
        print("🔍 Initializing Hybrid Candidate Search Engine...")
        self.load_all_data()
        self.build_search_index()
        self.build_tfidf_index()

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

    def extract_resume_text_for_tfidf(self, parsed_data: Dict) -> str:
        """Extract resume text optimized for keyword search"""
        text_parts = []
        
        # Add positions with emphasis on titles and companies
        if 'positions' in parsed_data:
            for position in parsed_data['positions']:
                # Job titles and companies are important for keyword matching
                if position.get('title'):
                    text_parts.append(position['title'])
                    text_parts.append(position['title'])  # Boost importance
                if position.get('org'):
                    text_parts.append(position['org'])
                    text_parts.append(position['org'])  # Boost importance
                if position.get('summary'):
                    text_parts.append(position['summary'])
                if position.get('location'):
                    text_parts.append(position['location'])
        
        # Add education with emphasis on institutions and degrees
        if 'schools' in parsed_data:
            for school in parsed_data['schools']:
                if school.get('org'):
                    text_parts.append(school['org'])
                    text_parts.append(school['org'])  # Boost importance
                if school.get('degree'):
                    text_parts.append(school['degree'])
                    text_parts.append(school['degree'])  # Boost importance
                if school.get('field'):
                    text_parts.append(school['field'])
                if school.get('summary'):
                    text_parts.append(school['summary'])
        
        # Add other fields
        for field in ['summary', 'skills', 'text']:
            if field in parsed_data and parsed_data[field]:
                if isinstance(parsed_data[field], str):
                    text_parts.append(parsed_data[field])
                elif isinstance(parsed_data[field], list):
                    text_parts.extend([str(item) for item in parsed_data[field]])
        
        return " ".join(text_parts)

    def build_search_index(self):
        """Build search index with both semantic embeddings and text"""
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
                    continue
                
                embedding_vector = resume_embedding['embedding']
                
                if not embedding_vector or len(embedding_vector) == 0:
                    continue
                
                try:
                    embedding_array = np.array(embedding_vector, dtype=np.float32)
                    
                    # Check for invalid values
                    if np.any(np.isnan(embedding_array)) or np.any(np.isinf(embedding_array)):
                        continue
                    
                    if np.all(embedding_array == 0):
                        continue
                    
                    # Normalize the embedding
                    embedding_norm = np.linalg.norm(embedding_array)
                    if embedding_norm > 0:
                        embedding_array = embedding_array / embedding_norm
                    else:
                        continue
                    
                except (ValueError, TypeError):
                    continue
                
                # Find corresponding resume text for TF-IDF
                resume_text = ""
                for resume in resume_data['resumes']:
                    if resume['id'] == resume_embedding['resume_id']:
                        parsed_data = resume.get('parsedData', {})
                        if parsed_data:
                            resume_text = self.extract_resume_text_for_tfidf(parsed_data)
                        break
                
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
                    'resume_text': resume_text,  # For TF-IDF
                    'embedding_dimensions': len(embedding_array),
                    'full_text_length': len(resume_text)
                }
                self.search_index.append(search_entry)
        
        print(f"✅ Built search index with {len(self.search_index)} valid resume embeddings")

    def build_tfidf_index(self):
        """Build TF-IDF index for keyword search"""
        print("🔤 Building TF-IDF keyword index...")
        
        if not self.search_index:
            print("❌ No search index available for TF-IDF")
            return
        
        # Extract resume texts
        self.resume_texts = [entry['resume_text'] for entry in self.search_index]
        
        # Create TF-IDF vectorizer with optimized parameters for resumes
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,  # Limit vocabulary size
            stop_words='english',  # Remove common English words
            ngram_range=(1, 2),  # Include both single words and bigrams
            min_df=1,  # Include terms that appear in at least 1 document
            max_df=0.8,  # Exclude terms that appear in more than 80% of documents
            lowercase=True,
            strip_accents='unicode'
        )
        
        # Fit and transform the resume texts
        try:
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.resume_texts)
            print(f"✅ Built TF-IDF index with {self.tfidf_matrix.shape[1]} features")
        except Exception as e:
            print(f"❌ Error building TF-IDF index: {e}")
            self.tfidf_matrix = None

    def preprocess_query(self, query: str) -> str:
        """Preprocess query for better case handling and matching"""
        # Handle common company/organization name patterns
        query = query.strip()
        
        # List of known companies/organizations that should be capitalized properly
        company_mappings = {
            'gyant': 'GYANT',
            'google': 'Google',
            'microsoft': 'Microsoft',
            'stanford': 'Stanford',
            'mit': 'MIT',
            'amazon': 'Amazon',
            'facebook': 'Facebook',
            'meta': 'Meta',
            'openai': 'OpenAI',
            'nvidia': 'NVIDIA',
            'apple': 'Apple',
            'tesla': 'Tesla',
            'uber': 'Uber',
            'airbnb': 'Airbnb',
            'linkedin': 'LinkedIn',
            'twitter': 'Twitter',
            'netflix': 'Netflix',
            'spotify': 'Spotify'
        }
        
        # Split query into words
        words = query.split()
        processed_words = []
        
        for word in words:
            word_lower = word.lower().strip('.,!?')
            
            # Check if it's a known company/organization
            if word_lower in company_mappings:
                # Replace with proper capitalization
                punctuation = ''.join(c for c in word if not c.isalnum())
                processed_words.append(company_mappings[word_lower] + punctuation)
            else:
                # Keep original word for other terms
                processed_words.append(word)
        
        processed_query = ' '.join(processed_words)
        return processed_query

    def create_query_embedding(self, query: str) -> Optional[np.ndarray]:
        """Create embedding for search query"""
        # Preprocess query for better matching
        processed_query = self.preprocess_query(query)
        
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=processed_query,
                encoding_format="float"
            )
            
            embedding = np.array(response.data[0].embedding, dtype=np.float32)
            
            # Normalize the query embedding
            embedding_norm = np.linalg.norm(embedding)
            if embedding_norm > 0:
                embedding = embedding / embedding_norm
            else:
                return None
            
            return embedding
        except Exception as e:
            print(f"❌ Error creating query embedding: {e}")
            return None

    def get_semantic_scores(self, query_embedding: np.ndarray) -> List[float]:
        """Get semantic similarity scores for all candidates"""
        semantic_scores = []
        
        for entry in self.search_index:
            try:
                candidate_embedding = entry['embedding']
                if candidate_embedding is None or len(candidate_embedding) == 0:
                    semantic_scores.append(0.0)
                    continue
                
                # Calculate cosine similarity using dot product (normalized vectors)
                similarity = np.dot(query_embedding, candidate_embedding)
                similarity = np.clip(similarity, -1.0, 1.0)
                semantic_scores.append(float(similarity))
                
            except Exception:
                semantic_scores.append(0.0)
        
        return semantic_scores

    def get_keyword_scores(self, query: str) -> List[float]:
        """Get TF-IDF keyword similarity scores for all candidates"""
        if self.tfidf_matrix is None or self.tfidf_vectorizer is None:
            return [0.0] * len(self.search_index)
        
        # Preprocess query for consistent matching
        processed_query = self.preprocess_query(query)
        
        try:
            # Transform query using fitted vectorizer
            query_vector = self.tfidf_vectorizer.transform([processed_query])
            
            # Calculate cosine similarity with all documents
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            return similarities.tolist()
            
        except Exception as e:
            print(f"⚠️  Error in keyword scoring: {e}")
            return [0.0] * len(self.search_index)

    def analyze_query(self, query: str) -> Tuple[float, float]:
        """Analyze query to determine optimal semantic vs keyword weighting"""
        query_lower = query.lower()
        
        # Technical terms that benefit from keyword search
        tech_terms = [
            'python', 'javascript', 'react', 'django', 'tensorflow', 'pytorch',
            'aws', 'kubernetes', 'docker', 'sql', 'nosql', 'mongodb',
            'machine learning', 'deep learning', 'nlp', 'computer vision',
            'stanford', 'mit', 'google', 'microsoft', 'facebook', 'amazon'
        ]
        
        # Role/concept terms that benefit from semantic search
        role_terms = [
            'senior', 'junior', 'lead', 'principal', 'director', 'manager',
            'startup', 'experience', 'team', 'leadership', 'strategy'
        ]
        
        tech_count = sum(1 for term in tech_terms if term in query_lower)
        role_count = sum(1 for term in role_terms if term in query_lower)
        
        # Adjust weights based on query characteristics
        if tech_count > role_count:
            # Technical query - increase keyword weight
            return 0.5, 0.5  # 50/50 split
        elif role_count > tech_count:
            # Role/concept query - increase semantic weight
            return 0.7, 0.3  # 70% semantic, 30% keyword
        else:
            # Balanced query - use config values
            return SEMANTIC_WEIGHT, KEYWORD_WEIGHT

    def hybrid_search(self, query: str, top_k: int = 5, min_similarity: float = 0.1, 
                     adaptive_weights: bool = False, debug_mode: bool = False) -> List[Dict]:
        """Perform hybrid search combining semantic and keyword approaches"""
        # Preprocess query and show what we're actually searching for
        processed_query = self.preprocess_query(query)
        
        if processed_query != query:
            print(f"🔍 Hybrid search for: '{query}' → '{processed_query}' (threshold: {min_similarity:.2f})")
        else:
            print(f"🔍 Hybrid search for: '{query}' (threshold: {min_similarity:.2f})")
        
        if not self.search_index:
            print("❌ No search index available")
            return []
        
        print(f"📊 Search index contains {len(self.search_index)} embeddings")
        
        # Determine weights
        if adaptive_weights:
            semantic_weight, keyword_weight = self.analyze_query(processed_query)
            print(f"🎛️  Adaptive weights: {semantic_weight:.1f} semantic + {keyword_weight:.1f} keyword")
        else:
            semantic_weight, keyword_weight = self.semantic_weight, self.keyword_weight
            print(f"🎛️  Fixed weights: {semantic_weight:.1f} semantic + {keyword_weight:.1f} keyword")
        
        # Get semantic scores (using original query since create_query_embedding handles preprocessing)
        query_embedding = self.create_query_embedding(query)
        if query_embedding is None:
            print("❌ Could not create query embedding")
            return []
        
        semantic_scores = self.get_semantic_scores(query_embedding)
        print(f"📈 Calculated {len(semantic_scores)} semantic scores")
        
        # Get keyword scores (using original query since get_keyword_scores handles preprocessing)
        keyword_scores = self.get_keyword_scores(query)
        print(f"🔤 Calculated {len(keyword_scores)} keyword scores")
        
        # Combine scores
        results = []
        for i, entry in enumerate(self.search_index):
            semantic_score = semantic_scores[i]
            keyword_score = keyword_scores[i]
            
            # Combined score
            combined_score = (semantic_weight * semantic_score) + (keyword_weight * keyword_score)
            
            if combined_score >= min_similarity:
                result = entry.copy()
                result['semantic_score'] = semantic_score
                result['keyword_score'] = keyword_score
                result['combined_score'] = combined_score
                results.append(result)
        
        # Sort by combined score
        results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        print(f"✅ Found {len(results)} matches above threshold {min_similarity:.2f}")
        
        # Debug output
        if debug_mode and results:
            print(f"\n🔍 Debug Mode - Top {min(5, len(results))} Results:")
            for i, result in enumerate(results[:5]):
                print(f"  {i+1}. {result['candidate_name']}")
                print(f"     Semantic: {result['semantic_score']:.3f} | Keyword: {result['keyword_score']:.3f} | Combined: {result['combined_score']:.3f}")
        
        return results[:top_k]

    def display_results(self, results: List[Dict], query: str):
        """Display search results with score breakdown"""
        if not results:
            print(f"\n❌ No matches found for '{query}'")
            print("Try lowering the similarity threshold or using different keywords")
            return
        
        print(f"\n🎯 Found {len(results)} matches for '{query}':")
        print("=" * 80)
        
        for i, result in enumerate(results, 1):
            combined_percent = result['combined_score'] * 100
            semantic_percent = result['semantic_score'] * 100
            keyword_percent = result['keyword_score'] * 100
            
            print(f"\n{i}. {result['candidate_name']} ({combined_percent:.1f}% match)")
            print(f"   📊 Scores: Semantic {semantic_percent:.1f}% | Keyword {keyword_percent:.1f}% | Combined {combined_percent:.1f}%")
            print(f"   📧 Email: {result['candidate_email'] or 'Not available'}")
            print(f"   📍 Location: {result['candidate_location'] or 'Not specified'}")
            print(f"   💼 Headline: {result['candidate_headline'] or 'Not specified'}")
            print(f"   📄 Resume: {result['resume_filename']}")
            print(f"   📝 Preview: {result['text_preview'][:200]}...")
            print(f"   🔗 Candidate ID: {result['candidate_id']}")
            print("-" * 80)

    def interactive_search(self):
        """Interactive search interface with hybrid options"""
        print("\n🔍 Hybrid Interactive Candidate Search")
        print("=" * 50)
        print("Enter search queries to find candidates using hybrid search")
        print("Special commands:")
        print("  - 'quit' or 'exit' to stop")
        print("  - 'debug on/off' to toggle debug mode")
        print("  - 'threshold X' to set similarity threshold (e.g., 'threshold 0.05')")
        print("  - 'weights X Y' to set semantic/keyword weights (e.g., 'weights 0.7 0.3')")
        print("  - 'adaptive on/off' to toggle adaptive weighting")
        print("  - 'stats' to show search engine statistics")
        print("")
        print("Examples:")
        print("  - 'Python developer'")
        print("  - 'Stanford graduate with AI experience'")
        print("  - 'CTO with startup background'")
        print("")
        
        debug_mode = False
        current_threshold = 0.1
        adaptive_weights = False
        
        print(f"🎛️  Current settings: threshold={current_threshold:.2f}, adaptive_weights={adaptive_weights}, debug={debug_mode}")
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
                
                if query.lower().startswith('weights '):
                    try:
                        parts = query.split(' ')
                        if len(parts) == 3:
                            sem_weight = float(parts[1])
                            key_weight = float(parts[2])
                            if abs(sem_weight + key_weight - 1.0) < 0.01:  # Should sum to ~1.0
                                self.semantic_weight = sem_weight
                                self.keyword_weight = key_weight
                                adaptive_weights = False  # Disable adaptive when manually set
                                print(f"🎛️  Weights set to: {sem_weight:.1f} semantic + {key_weight:.1f} keyword")
                            else:
                                print("❌ Weights should sum to 1.0")
                        else:
                            print("❌ Use format: 'weights 0.6 0.4'")
                    except ValueError:
                        print("❌ Invalid weight values")
                    continue
                
                if query.lower().startswith('adaptive '):
                    mode = query.lower().split(' ', 1)[1]
                    if mode in ['on', 'true', '1']:
                        adaptive_weights = True
                        print("🎛️  Adaptive weighting: ON")
                    elif mode in ['off', 'false', '0']:
                        adaptive_weights = False
                        print("🎛️  Adaptive weighting: OFF")
                    else:
                        print("❌ Use 'adaptive on' or 'adaptive off'")
                    continue
                
                if query.lower() == 'stats':
                    self.get_search_statistics()
                    continue
                
                # Perform hybrid search
                results = self.hybrid_search(
                    query, 
                    min_similarity=current_threshold, 
                    adaptive_weights=adaptive_weights,
                    debug_mode=debug_mode
                )
                self.display_results(results, query)
                
                print(f"\n{'-' * 50}")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error during search: {e}")

    def get_search_statistics(self):
        """Display search engine statistics"""
        print("\n📊 Hybrid Search Engine Statistics")
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
        
        if self.tfidf_matrix is not None:
            print(f"TF-IDF vocabulary size: {self.tfidf_matrix.shape[1]}")
            print(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")

def main():
    """Main function"""
    print("🚀 Starting Hybrid Candidate Search Engine")
    print("=" * 60)
    
    # Check for OpenAI API key
    openai_api_key = os.getenv('OPEN_AI_API_KEY')
    if not openai_api_key:
        print("❌ Error: OPEN_AI_API_KEY environment variable not set")
        print("Please set it in your .env file or environment")
        return
    
    try:
        # Initialize hybrid search engine
        search_engine = HybridCandidateSearch(openai_api_key)
        
        # Display statistics
        search_engine.get_search_statistics()
        
        # Start interactive search
        search_engine.interactive_search()
    
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please run main.py first to extract resume data and create embeddings.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()