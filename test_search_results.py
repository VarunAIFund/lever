#!/usr/bin/env python3
"""
Test Suite for Resume Search Engine
Validates search results against expected candidates and similarity thresholds
"""

import os
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time

# Add current directory to path to import our modules
sys.path.append(str(Path(__file__).parent))

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

# Import our search engine
from candidate_search import CandidateSearchEngine

@dataclass
class TestQuery:
    query: str
    expected_candidates: List[str]  # Candidate names that should appear in results
    min_similarity: float
    expected_top_candidate: Optional[str] = None  # Name of candidate that should rank #1
    description: str = ""

@dataclass
class TestResult:
    query: str
    passed: bool
    actual_candidates: List[str]
    expected_candidates: List[str]
    top_similarity: float
    message: str

class SearchEngineValidator:
    def __init__(self):
        print("🚀 Initializing Search Engine Validator")
        print("=" * 60)
        
        # Check for OpenAI API key
        openai_api_key = os.getenv('OPEN_AI_API_KEY')
        if not openai_api_key:
            raise ValueError("❌ Error: OPEN_AI_API_KEY environment variable not set")
        
        # Initialize search engine
        self.search_engine = CandidateSearchEngine(openai_api_key)
        self.test_results = []
        
        print("✅ Search engine initialized successfully")
        
    def run_query_test(self, test_query: TestQuery, top_k: int = 5) -> TestResult:
        """Run a single test query and validate results"""
        print(f"\n🔍 Testing: '{test_query.query}'")
        
        # Perform search
        results = self.search_engine.search(
            test_query.query, 
            top_k=top_k, 
            min_similarity=test_query.min_similarity,
            debug_mode=False
        )
        
        # Extract candidate names from results
        actual_candidates = [result['candidate_name'] for result in results]
        top_similarity = results[0]['similarity_score'] if results else 0.0
        
        # Check if expected candidates are in results
        found_expected = []
        for expected in test_query.expected_candidates:
            if any(expected.lower() in candidate.lower() for candidate in actual_candidates):
                found_expected.append(expected)
        
        # Check if top candidate matches expectation
        top_candidate_correct = True
        if test_query.expected_top_candidate and results:
            top_actual = results[0]['candidate_name']
            top_candidate_correct = test_query.expected_top_candidate.lower() in top_actual.lower()
        
        # Determine if test passed
        expected_found_ratio = len(found_expected) / len(test_query.expected_candidates) if test_query.expected_candidates else 1.0
        passed = expected_found_ratio >= 0.5 and top_candidate_correct  # At least 50% of expected candidates found
        
        # Create result message
        if passed:
            message = f"✅ PASS - Found {len(found_expected)}/{len(test_query.expected_candidates)} expected candidates"
        else:
            message = f"❌ FAIL - Found {len(found_expected)}/{len(test_query.expected_candidates)} expected candidates"
        
        if test_query.expected_top_candidate:
            if top_candidate_correct:
                message += f", top candidate correct"
            else:
                message += f", expected top: {test_query.expected_top_candidate}, got: {actual_candidates[0] if actual_candidates else 'None'}"
        
        print(f"   Results: {actual_candidates[:3]}...")
        print(f"   Top similarity: {top_similarity:.3f}")
        print(f"   {message}")
        
        return TestResult(
            query=test_query.query,
            passed=passed,
            actual_candidates=actual_candidates,
            expected_candidates=test_query.expected_candidates,
            top_similarity=top_similarity,
            message=message
        )
    
    def run_individual_candidate_tests(self) -> List[TestResult]:
        """Test queries targeted at specific candidates"""
        print("\n" + "="*60)
        print("🎯 INDIVIDUAL CANDIDATE TESTS")
        print("="*60)
        
        individual_tests = [
            # Amol Kelkar (AI/CTO/Entrepreneur)
            TestQuery("CTO with AI startup experience", ["Amol Kelkar"], 0.1, "Amol Kelkar"),
            TestQuery("Co-founder with machine learning background", ["Amol Kelkar"], 0.1),
            TestQuery("Senior engineer with patents in AI", ["Amol Kelkar"], 0.1),
            TestQuery("AGI research scientist", ["Amol Kelkar"], 0.1),
            TestQuery("Microsoft veteran with entrepreneurship experience", ["Amol Kelkar"], 0.1),
            
            # Eitan Anzenberg (Data Science Leader)
            TestQuery("Chief Data Scientist with deep learning experience", ["Eitan Anzenberg"], 0.1, "Eitan Anzenberg"),
            TestQuery("Machine learning leader with computer vision background", ["Eitan Anzenberg"], 0.1),
            TestQuery("Data science director with patents", ["Eitan Anzenberg"], 0.1),
            TestQuery("NLP expert with production experience", ["Eitan Anzenberg"], 0.1),
            TestQuery("PhD in physics with ML background", ["Eitan Anzenberg"], 0.1),
            
            # Hakan Gunturkun (AI/ML Engineer)
            TestQuery("Head of AI with generative models experience", ["Hakan Gunturkun"], 0.1, "Hakan Gunturkun"),
            TestQuery("Co-founder with LLM background", ["Hakan Gunturkun"], 0.1),
            TestQuery("AI engineer with diffusion models", ["Hakan Gunturkun"], 0.1),
            TestQuery("PhD with startup experience", ["Hakan Gunturkun"], 0.1),
            TestQuery("Machine learning researcher with production experience", ["Hakan Gunturkun"], 0.1),
            
            # Kirill Kireyev (AI Entrepreneur/NLP)
            TestQuery("AI entrepreneur with NLP background", ["Kirill Kireyev"], 0.1, "Kirill Kireyev"),
            TestQuery("CTO with educational technology experience", ["Kirill Kireyev"], 0.1),
            TestQuery("Natural language processing expert", ["Kirill Kireyev"], 0.1),
            TestQuery("PhD with startup founder experience", ["Kirill Kireyev"], 0.1),
            TestQuery("Machine learning consultant", ["Kirill Kireyev"], 0.1),
            
            # Majed Itani (Engineering Leader/CTO)
            TestQuery("CTO with team building experience", ["Majed Itani"], 0.1),
            TestQuery("Engineering leader with CRM background", ["Majed Itani"], 0.1),
            TestQuery("Co-founder with enterprise software experience", ["Majed Itani"], 0.1),
            TestQuery("VP of Engineering with patents", ["Majed Itani"], 0.1),
            TestQuery("Technical leader with recruiting experience", ["Majed Itani"], 0.1),
            
            # Mohamed El-Geish (AI Director/Academic)
            TestQuery("Director of AI with academic background", ["Mohamed El-Geish"], 0.1, "Mohamed El-Geish"),
            TestQuery("Stanford teaching assistant with industry experience", ["Mohamed El-Geish"], 0.1),
            TestQuery("AI leader with contact center experience", ["Mohamed El-Geish"], 0.1),
            TestQuery("Machine learning expert with Cisco background", ["Mohamed El-Geish"], 0.1),
            TestQuery("Co-founder with deep learning experience", ["Mohamed El-Geish"], 0.1),
            
            # Vid Jain (CEO/AI Platform)
            TestQuery("CEO with AI platform experience", ["Vid Jain"], 0.1, "Vid Jain"),
            TestQuery("Founder with machine learning operations background", ["Vid Jain"], 0.1),
            TestQuery("Enterprise AI deployment expert", ["Vid Jain"], 0.1),
            TestQuery("Physics PhD with business experience", ["Vid Jain"], 0.1),
            TestQuery("Fintech executive with ML background", ["Vid Jain"], 0.1),
            
            # Yefei Peng (Engineering Leader/ML)
            TestQuery("Senior Director with machine learning experience", ["Yefei Peng"], 0.1, "Yefei Peng"),
            TestQuery("Engineering manager with deep learning background", ["Yefei Peng"], 0.1),
            TestQuery("Tech lead with recommendation systems experience", ["Yefei Peng"], 0.1),
            TestQuery("Google veteran with ML production experience", ["Yefei Peng"], 0.1),
            TestQuery("PhD with large-scale systems experience", ["Yefei Peng"], 0.1),
            
            # Yue Ning (CTO/NLP Startup)
            TestQuery("CTO with NLP startup experience", ["Yue Ning"], 0.1, "Yue Ning"),
            TestQuery("Co-founder with text analytics background", ["Yue Ning"], 0.1),
            TestQuery("Natural language processing expert", ["Yue Ning"], 0.1),
            TestQuery("Startup founder with AI focus", ["Yue Ning"], 0.1),
            TestQuery("Engineering manager with NLP products", ["Yue Ning"], 0.1),
            
            # Zackary Long (Engineering Director/Fintech)
            TestQuery("Director of Engineering with fintech experience", ["Zackary Long"], 0.1, "Zackary Long"),
            TestQuery("Engineering leader with machine learning background", ["Zackary Long"], 0.1),
            TestQuery("Startup CTO with scalability experience", ["Zackary Long"], 0.1),
            TestQuery("Technical leader with data pipelines experience", ["Zackary Long"], 0.1),
            TestQuery("Engineering manager with AI recommendations", ["Zackary Long"], 0.1),
        ]
        
        results = []
        for test in individual_tests:
            result = self.run_query_test(test)
            results.append(result)
            self.test_results.append(result)
            time.sleep(0.1)  # Rate limiting
        
        return results
    
    def run_cross_candidate_tests(self) -> List[TestResult]:
        """Test queries that should return multiple specific candidates"""
        print("\n" + "="*60)
        print("🔀 CROSS-CANDIDATE COMPARISON TESTS")
        print("="*60)
        
        cross_tests = [
            TestQuery(
                "AI startup CTO", 
                ["Amol Kelkar", "Hakan Gunturkun", "Yue Ning"], 
                0.1,
                description="Should prioritize AI startup CTOs"
            ),
            TestQuery(
                "Machine learning research", 
                ["Eitan Anzenberg", "Mohamed El-Geish", "Hakan Gunturkun"], 
                0.1,
                description="Should find ML researchers"
            ),
            TestQuery(
                "PhD with industry experience", 
                ["Eitan Anzenberg", "Kirill Kireyev", "Mohamed El-Geish", "Vid Jain", "Yefei Peng"], 
                0.1,
                description="Should find PhD holders in industry"
            ),
            TestQuery(
                "Enterprise software leader", 
                ["Majed Itani", "Yefei Peng", "Zackary Long"], 
                0.1,
                description="Should find enterprise software leaders"
            ),
            TestQuery(
                "Natural language processing expert", 
                ["Kirill Kireyev", "Yue Ning", "Mohamed El-Geish"], 
                0.1,
                description="Should find NLP experts"
            ),
        ]
        
        results = []
        for test in cross_tests:
            print(f"\n📋 {test.description}")
            result = self.run_query_test(test, top_k=8)  # Get more results for multi-candidate tests
            results.append(result)
            self.test_results.append(result)
            time.sleep(0.1)  # Rate limiting
        
        return results
    
    def run_edge_case_tests(self) -> List[TestResult]:
        """Test queries that should return low similarity scores"""
        print("\n" + "="*60)
        print("🚫 EDGE CASE TESTS")
        print("="*60)
        
        edge_tests = [
            TestQuery("Blockchain developer", [], 0.3, description="Should have low similarity - not in dataset"),
            TestQuery("Frontend React developer", [], 0.3, description="Should have low similarity - not primary expertise"),
            TestQuery("Marketing manager", [], 0.3, description="Should have low similarity - different field"),
            TestQuery("Sales representative", [], 0.3, description="Should have low similarity - different field"),
            TestQuery("Graphic designer", [], 0.3, description="Should have low similarity - different field"),
        ]
        
        results = []
        for test in edge_tests:
            print(f"\n🚫 {test.description}")
            result = self.run_query_test(test)
            # For edge cases, we expect low similarity scores or no results
            if result.top_similarity < test.min_similarity or not result.actual_candidates:
                result.passed = True
                result.message = f"✅ PASS - Low similarity as expected ({result.top_similarity:.3f})"
            else:
                result.passed = False
                result.message = f"❌ FAIL - Unexpected high similarity ({result.top_similarity:.3f})"
            
            print(f"   {result.message}")
            results.append(result)
            self.test_results.append(result)
            time.sleep(0.1)  # Rate limiting
        
        return results
    
    def generate_report(self) -> str:
        """Generate a comprehensive test report"""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result.passed)
        failed_tests = total_tests - passed_tests
        
        report = []
        report.append("📊 SEARCH ENGINE TEST REPORT")
        report.append("=" * 60)
        report.append(f"Total Tests: {total_tests}")
        report.append(f"Passed: {passed_tests} ({passed_tests/total_tests*100:.1f}%)")
        report.append(f"Failed: {failed_tests} ({failed_tests/total_tests*100:.1f}%)")
        report.append("")
        
        if failed_tests > 0:
            report.append("❌ FAILED TESTS:")
            report.append("-" * 40)
            for result in self.test_results:
                if not result.passed:
                    report.append(f"Query: '{result.query}'")
                    report.append(f"Message: {result.message}")
                    report.append(f"Expected: {result.expected_candidates}")
                    report.append(f"Actual: {result.actual_candidates[:3]}")
                    report.append("")
        
        # Summary by category
        individual_results = [r for r in self.test_results if len(r.expected_candidates) <= 1]
        cross_results = [r for r in self.test_results if len(r.expected_candidates) > 1 and r.expected_candidates]
        edge_results = [r for r in self.test_results if not r.expected_candidates]
        
        report.append("📈 RESULTS BY CATEGORY:")
        report.append("-" * 40)
        if individual_results:
            individual_passed = sum(1 for r in individual_results if r.passed)
            report.append(f"Individual Candidate Tests: {individual_passed}/{len(individual_results)} passed")
        
        if cross_results:
            cross_passed = sum(1 for r in cross_results if r.passed)
            report.append(f"Cross-Candidate Tests: {cross_passed}/{len(cross_results)} passed")
        
        if edge_results:
            edge_passed = sum(1 for r in edge_results if r.passed)
            report.append(f"Edge Case Tests: {edge_passed}/{len(edge_results)} passed")
        
        return "\n".join(report)
    
    def run_all_tests(self):
        """Run the complete test suite"""
        print("🧪 Starting Complete Test Suite")
        print("This may take a few minutes due to API rate limiting...")
        
        start_time = time.time()
        
        try:
            # Run all test categories
            self.run_individual_candidate_tests()
            self.run_cross_candidate_tests()
            self.run_edge_case_tests()
            
            # Generate and display report
            report = self.generate_report()
            print("\n" + report)
            
            # Save report to file
            report_file = Path("test_report.txt")
            with open(report_file, 'w') as f:
                f.write(report)
            
            end_time = time.time()
            print(f"\n⏱️  Total test time: {end_time - start_time:.1f} seconds")
            print(f"📄 Full report saved to: {report_file}")
            
        except Exception as e:
            print(f"❌ Test suite failed with error: {e}")
            raise

def main():
    """Main function to run tests"""
    print("🚀 Resume Search Engine Test Suite")
    print("=" * 60)
    
    try:
        validator = SearchEngineValidator()
        validator.run_all_tests()
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please run main.py first to extract resume data and create embeddings.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()