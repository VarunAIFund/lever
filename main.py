#!/usr/bin/env python3
"""
Main script to orchestrate Lever resume extraction and embeddings creation
"""

import sys
import os
from pathlib import Path

# Add current directory to path so we can import our modules
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from lever_extractor import LeverResumeExtractor, load_env_file
from create_embeddings import ResumeEmbeddingsCreator

def main():
    """Main orchestration function"""
    print("🚀 Starting Lever Resume Processing Pipeline")
    print("=" * 60)
    
    # Load environment variables
    load_env_file()
    
    # Check for required API keys
    lever_api_key = os.getenv('LEVER_API_KEY')
    openai_api_key = os.getenv('OPEN_AI_API_KEY')
    
    if not lever_api_key:
        print("❌ Error: LEVER_API_KEY environment variable not set")
        print("Please set it in your .env file or environment")
        return
    
    if not openai_api_key:
        print("⚠️  Warning: OPEN_AI_API_KEY environment variable not set")
        print("Only resume extraction will be performed (no embeddings)")
        print("Set OPEN_AI_API_KEY in your .env file to enable embeddings\n")
    
    # Step 1: Extract resume data from Lever API
    print("\n📋 STEP 1: Extracting resume data from Lever API")
    print("-" * 50)
    
    try:
        extractor = LeverResumeExtractor(lever_api_key)
        extractor.run()
    except Exception as e:
        print(f"❌ Error during resume extraction: {e}")
        return
    
    # Step 2: Create embeddings (if OpenAI API key is available)
    if openai_api_key:
        print("\n🔮 STEP 2: Creating embeddings from resume data")
        print("-" * 50)
        
        try:
            # Check if OpenAI is available
            try:
                import openai
                creator = ResumeEmbeddingsCreator(openai_api_key)
                creator.run()
            except ImportError:
                print("❌ Error: OpenAI library not installed")
                print("Install with: pip install openai")
                print("Skipping embeddings creation.")
        except Exception as e:
            print(f"❌ Error during embeddings creation: {e}")
            print("Resume data was still extracted successfully.")
    
    print("\n" + "=" * 60)
    print("✅ Pipeline complete!")
    
    # Summary
    data_dir = Path("data")
    files_created = []
    if (data_dir / "candidates.json").exists():
        files_created.append("candidates.json")
    if (data_dir / "resumes_data.json").exists():
        files_created.append("resumes_data.json")
    if (data_dir / "resume_embeddings.json").exists():
        files_created.append("resume_embeddings.json")
    
    if files_created:
        print(f"📁 Files created: {', '.join(files_created)}")
    
    resume_files_dir = Path("resume_files")
    if resume_files_dir.exists():
        pdf_count = len(list(resume_files_dir.glob("*.pdf")))
        if pdf_count > 0:
            print(f"📄 Downloaded {pdf_count} PDF files")


if __name__ == "__main__":
    main()