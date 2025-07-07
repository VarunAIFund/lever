#!/usr/bin/env python3
"""
Lever API Resume Extractor
Fetches candidates from Lever API and downloads their resumes
"""

import os
import json
import requests
import time
from pathlib import Path
from typing import Dict, List, Optional
import base64
from urllib.parse import urlparse
import mimetypes

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

class LeverResumeExtractor:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.lever.co/v1"
        self.session = requests.Session()
        self.session.auth = (api_key, '')
        self.rate_limit_delay = 0.6  # ~100 requests per minute
        
        # Create directory structure
        self.data_dir = Path("data")
        self.resume_files_dir = Path("resume_files")
        self.data_dir.mkdir(exist_ok=True)
        self.resume_files_dir.mkdir(exist_ok=True)
        
        print(f"📁 Created directories: {self.data_dir}, {self.resume_files_dir}")

    def make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make API request with rate limiting and error handling"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            print(f"🔍 Making request to: {endpoint}")
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            # Rate limiting
            time.sleep(self.rate_limit_delay)
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Error making request to {endpoint}: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response status: {e.response.status_code}")
                print(f"Response body: {e.response.text[:200]}...")
            return {}

    def download_file(self, url: str, filename: str) -> bool:
        """Download file from URL"""
        try:
            print(f"📥 Downloading: {filename}")
            response = self.session.get(url, stream=True)
            response.raise_for_status()
            
            filepath = self.resume_files_dir / filename
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"✅ Downloaded: {filepath}")
            time.sleep(self.rate_limit_delay)
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Error downloading {filename}: {e}")
            return False

    def fetch_candidates(self, limit: int = 10) -> List[Dict]:
        """Fetch candidates from Lever API"""
        print(f"🔍 Fetching {limit} candidates...")
        
        params = {
            'limit': limit
        }
        
        data = self.make_request("/opportunities", params)
        
        if not data or 'data' not in data:
            print("❌ No candidates data received")
            return []
        
        candidates = data['data']
        print(f"✅ Found {len(candidates)} candidates")
        return candidates

    def fetch_candidate_resumes(self, candidate_id: str) -> List[Dict]:
        """Fetch resumes for a specific candidate"""
        print(f"📄 Fetching resumes for candidate: {candidate_id}")
        
        endpoint = f"/opportunities/{candidate_id}/resumes"
        data = self.make_request(endpoint)
        
        if not data or 'data' not in data:
            print(f"⚠️  No resumes found for candidate: {candidate_id}")
            return []
        
        resumes = data['data']
        print(f"✅ Found {len(resumes)} resume(s) for candidate: {candidate_id}")
        return resumes

    def process_candidate_resumes(self, candidate: Dict) -> Dict:
        """Process all resumes for a candidate"""
        candidate_id = candidate['id']
        candidate_name = candidate.get('name', 'Unknown')
        
        print(f"\n👤 Processing candidate: {candidate_name} ({candidate_id})")
        
        resumes = self.fetch_candidate_resumes(candidate_id)
        
        candidate_resume_data = {
            'candidate_id': candidate_id,
            'candidate_name': candidate_name,
            'resumes': [],
            'downloaded_files': []
        }
        
        for i, resume in enumerate(resumes, 1):
            resume_id = resume['id']
            resume_filename = resume.get('filename', f"resume_{i}")
            
            # Get file information
            file_info = resume.get('file', {})
            actual_filename = file_info.get('name', resume_filename)
            file_extension = file_info.get('ext', '.pdf')  # Default to .pdf if no extension
            
            # Create safe filename with proper extension
            base_safe_filename = f"{candidate_name}_{resume_filename}".replace(' ', '_')
            base_safe_filename = ''.join(c for c in base_safe_filename if c.isalnum() or c in '._-')
            safe_filename = f"{base_safe_filename}{file_extension}"
            
            resume_data = {
                'id': resume_id,
                'filename': actual_filename,
                'safe_filename': safe_filename,
                'createdAt': resume.get('createdAt'),
                'file_info': file_info,
                'parsedData': resume.get('parsedData', {})
            }
            
            # Try to download resume file using the correct download endpoint
            download_url = f"{self.base_url}/opportunities/{candidate_id}/resumes/{resume_id}/download"
            if self.download_file(download_url, safe_filename):
                candidate_resume_data['downloaded_files'].append(safe_filename)
                resume_data['downloaded'] = True
            else:
                resume_data['downloaded'] = False
            
            candidate_resume_data['resumes'].append(resume_data)
        
        return candidate_resume_data

    def save_data(self, candidates: List[Dict], resume_data: List[Dict]):
        """Save data to JSON files"""
        print(f"\n💾 Saving data to files...")
        
        # Save candidates data
        candidates_file = self.data_dir / "candidates.json"
        with open(candidates_file, 'w') as f:
            json.dump(candidates, f, indent=2)
        print(f"✅ Saved candidates data: {candidates_file}")
        
        # Save resumes data
        resumes_file = self.data_dir / "resumes_data.json"
        with open(resumes_file, 'w') as f:
            json.dump(resume_data, f, indent=2)
        print(f"✅ Saved resumes data: {resumes_file}")

    def run(self):
        """Main execution function"""
        print("🚀 Starting Lever Resume Extractor")
        print("=" * 50)
        
        # Fetch candidates
        candidates = self.fetch_candidates(10)
        if not candidates:
            print("❌ No candidates found. Exiting.")
            return
        
        # Process each candidate's resumes
        all_resume_data = []
        for candidate in candidates:
            try:
                candidate_resume_data = self.process_candidate_resumes(candidate)
                all_resume_data.append(candidate_resume_data)
            except Exception as e:
                print(f"❌ Error processing candidate {candidate.get('id', 'unknown')}: {e}")
                continue
        
        # Save all data
        self.save_data(candidates, all_resume_data)
        
        print("\n" + "=" * 50)
        print("✅ Resume extraction complete!")
        print(f"📊 Processed {len(candidates)} candidates")
        print(f"📄 Found {sum(len(data['resumes']) for data in all_resume_data)} resumes")
        print(f"💾 Downloaded {sum(len(data['downloaded_files']) for data in all_resume_data)} files")


def main():
    """Main entry point"""
    # Check for API key
    lever_api_key = os.getenv('LEVER_API_KEY')
    if not lever_api_key:
        print("❌ Error: LEVER_API_KEY environment variable not set")
        print("Please set it in your .env file or environment")
        return
    
    # Create extractor and run
    extractor = LeverResumeExtractor(lever_api_key)
    extractor.run()


if __name__ == "__main__":
    main()