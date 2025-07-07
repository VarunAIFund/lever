#!/usr/bin/env python3
"""
Step 1: Test Environment Setup
"""
import os
from pathlib import Path

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

# Test environment loading
load_env_file()

# Test OpenAI import
try:
    import openai
    OPENAI_AVAILABLE = True
    print("✅ OpenAI library available")
except ImportError:
    OPENAI_AVAILABLE = False
    print("❌ OpenAI library not available")

# Test API key
openai_api_key = os.getenv('OPEN_AI_API_KEY')
if openai_api_key:
    print("✅ OpenAI API key found")
else:
    print("❌ OpenAI API key not found")

print("Step 1 complete!")


#!/usr/bin/env python3
"""
Step 2: Test Data Loading
"""
import json
from pathlib import Path

def test_load_resume_data():
    """Test loading resume data"""
    data_dir = Path("data")
    resumes_file = data_dir / "resumes_data.json"
    
    if not resumes_file.exists():
        print(f"❌ Resume data file not found: {resumes_file}")
        return None
    
    with open(resumes_file, 'r') as f:
        resume_data = json.load(f)
    
    print(f"✅ Loaded resume data for {len(resume_data)} candidates")
    
    # Show structure of first candidate
    if resume_data:
        first_candidate = resume_data[0]
        print(f"First candidate: {first_candidate['candidate_name']}")
        print(f"Number of resumes: {len(first_candidate['resumes'])}")
        
        if first_candidate['resumes']:
            first_resume = first_candidate['resumes'][0]
            print(f"First resume filename: {first_resume.get('filename', 'unknown')}")
            print(f"Has parsed data: {'parsedData' in first_resume}")
    
    return resume_data

# Test data loading
resume_data = test_load_resume_data()
#print(resume_data)
print("Step 2 complete!")


#!/usr/bin/env python3
"""
Step 3: Test Text Extraction
"""
import json
from pathlib import Path
from typing import Dict

def extract_resume_text_for_embeddings(parsed_data: Dict) -> str:
    """Extract and format resume text from parsed data for embeddings"""
    text_parts = []
    
    # Add positions/work experience
    if 'positions' in parsed_data:
        for position in parsed_data['positions']:
            parts = []
            if position.get('title'):
                parts.append(f"Title: {position['title']}")
            if position.get('org'):
                parts.append(f"Company: {position['org']}")
            if position.get('summary'):
                parts.append(f"Summary: {position['summary']}")
            if position.get('location'):
                parts.append(f"Location: {position['location']}")
            
            # Add date info
            if position.get('start'):
                start_date = position['start']
                if isinstance(start_date, dict):
                    start_str = f"{start_date.get('month', '')}/{start_date.get('year', '')}"
                    parts.append(f"Start: {start_str}")
            if position.get('end'):
                end_date = position['end']
                if isinstance(end_date, dict):
                    end_str = f"{end_date.get('month', '')}/{end_date.get('year', '')}"
                    parts.append(f"End: {end_str}")
            
            if parts:
                text_parts.append("POSITION: " + " | ".join(parts))
    
    # Add education/schools
    if 'schools' in parsed_data:
        for school in parsed_data['schools']:
            parts = []
            if school.get('org'):
                parts.append(f"School: {school['org']}")
            if school.get('degree'):
                parts.append(f"Degree: {school['degree']}")
            if school.get('field'):
                parts.append(f"Field: {school['field']}")
            if school.get('summary'):
                parts.append(f"Summary: {school['summary']}")
            
            if parts:
                text_parts.append("EDUCATION: " + " | ".join(parts))
    
    # Add any other text fields
    for field in ['summary', 'skills', 'text']:
        if field in parsed_data and parsed_data[field]:
            if isinstance(parsed_data[field], str):
                text_parts.append(f"{field.upper()}: {parsed_data[field]}")
            elif isinstance(parsed_data[field], list):
                text_parts.append(f"{field.upper()}: {' '.join(str(item) for item in parsed_data[field])}")
    output = "\n\n".join(text_parts)
    return output

def test_text_extraction():
    """Test text extraction on sample data"""
    data_dir = Path("data")
    resumes_file = data_dir / "resumes_data.json"
    
    with open(resumes_file, 'r') as f:
        resume_data = json.load(f)
    
    # Test on first candidate's first resume
    second_candidate = resume_data[1]
    #print(second_candidate)
    first_resume = second_candidate['resumes'][0]
    parsed_data = first_resume.get('parsedData', {})
    #print("Parsed Data: " + str(parsed_data))
    
    if parsed_data:
        extracted_text = extract_resume_text_for_embeddings(parsed_data)
        print(f"✅ Extracted text for {second_candidate['candidate_name']}")
        print(f"Text length: {len(extracted_text)} characters")
        print(f"Preview:\n{extracted_text[:500]}...")
        print(extracted_text)
    else:
        print("❌ No parsed data found")

# Test text extraction
test_text_extraction()
print("Step 3 complete!")



#!/usr/bin/env python3
"""
Step 4: Test Single Embedding Creation
"""
import os
import json
import openai
from pathlib import Path

# Load environment
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

def test_single_embedding():
    """Test creating a single embedding"""
    openai_api_key = os.getenv('OPEN_AI_API_KEY')
    if not openai_api_key:
        print("❌ No API key found")
        return
    
    client = openai.OpenAI(api_key=openai_api_key)
    
    # Test with simple text
    test_text = "Software Engineer with Python experience at Google"
    
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=test_text,
            encoding_format="float"
        )
        
        embedding_vector = response.data[0].embedding
        print(f"✅ Created embedding successfully!")
        print(f"Vector dimensions: {len(embedding_vector)}")
        print(f"First 5 values: {embedding_vector[:5]}")
        
    except Exception as e:
        print(f"❌ Error creating embedding: {e}")

# Test single embedding
#test_single_embedding()
print("Step 4 complete!")





#!/usr/bin/env python3
"""
Step 5: Test Full Pipeline (Limited to 2 candidates)
"""
import os
import json
import time
import openai
from pathlib import Path
from typing import Dict, List

# Load environment
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


def test_limited_pipeline():
    """Test full pipeline with limited data"""
    openai_api_key = os.getenv('OPEN_AI_API_KEY')
    if not openai_api_key:
        print("❌ No API key found")
        return
    
    client = openai.OpenAI(api_key=openai_api_key)
    data_dir = Path("data")
    resumes_file = data_dir / "resumes_data.json"
    
    with open(resumes_file, 'r') as f:
        resume_data = json.load(f)
    
    # Limit to first 2 candidates for testing
    limited_data = resume_data[:10]
    print(f"🧪 Testing with {len(limited_data)} candidates")
    
    embeddings_data = {}
    
    for candidate_data in limited_data:
        candidate_id = candidate_data['candidate_id']
        candidate_name = candidate_data['candidate_name']
        
        print(f"🔮 Processing: {candidate_name}")
        
        candidate_embeddings = {
            'candidate_id': candidate_id,
            'candidate_name': candidate_name,
            'resume_embeddings': []
        }
        
        for resume in candidate_data['resumes']:
            parsed_data = resume.get('parsedData', {})
            
            if not parsed_data:
                print(f"  ⚠️  No parsed data for resume")
                continue
            
            resume_text = extract_resume_text_for_embeddings(parsed_data)
            
            if not resume_text.strip():
                print(f"  ⚠️  No text extracted")
                continue
            
            try:
                response = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=resume_text,
                    encoding_format="float"
                )
                
                embedding_vector = response.data[0].embedding
                
                candidate_embeddings['resume_embeddings'].append({
                    'resume_id': resume['id'],
                    'filename': resume['filename'],
                    'text_preview': resume_text[:200] + "..." if len(resume_text) > 200 else resume_text,
                    'embedding': embedding_vector,
                    'embedding_dimensions': len(embedding_vector)
                })
                
                print(f"  ✅ Created embedding ({len(embedding_vector)} dimensions)")
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
                continue
        
        embeddings_data[candidate_id] = candidate_embeddings
    
    # Save test results
    test_file = data_dir / "test_embeddings.json"
    with open(test_file, 'w') as f:
        json.dump(embeddings_data, f, indent=2)
    
    print(f"✅ Test complete! Saved to {test_file}")
    total_embeddings = sum(len(data['resume_embeddings']) for data in embeddings_data.values())
    print(f"🔮 Created {total_embeddings} test embeddings")

# Test limited pipeline
test_limited_pipeline()
print("Step 5 complete!")
