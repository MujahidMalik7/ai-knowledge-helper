"""
Test script for AI Knowledge Helper API
Run this after starting the FastAPI server
"""

import requests
import json
import sys
from pathlib import Path

BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test if API is running"""
    print("\n" + "="*60)
    print("1. Testing Health Check...")
    print("="*60)
    
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            print("✅ API is running!")
            print(json.dumps(response.json(), indent=2))
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Is the server running?")
        print("Start it with: python main.py")
        return False

def test_upload(file_path: str):
    """Test document upload"""
    print("\n" + "="*60)
    print("2. Testing Document Upload...")
    print("="*60)
    
    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        return False
    
    try:
        with open(file_path, 'rb') as f:
            files = {'file': (Path(file_path).name, f)}
            response = requests.post(f"{BASE_URL}/upload", files=files)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Upload successful!")
            print(f"Filename: {result['filename']}")
            print(f"Chunks created: {result['chunks_created']}")
            print(f"Summary: {result['summary'][:200]}...")
            return True
        else:
            print(f"❌ Upload failed: {response.status_code}")
            print(response.text)
            return False
    
    except Exception as e:
        print(f"❌ Upload error: {str(e)}")
        return False

def test_ask_question(question: str):
    """Test question answering"""
    print("\n" + "="*60)
    print("3. Testing Question Answering...")
    print("="*60)
    print(f"Question: {question}")
    
    try:
        response = requests.post(
            f"{BASE_URL}/ask",
            json={"question": question, "top_k": 5}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ Answer received!")
            print(f"\nAnswer:\n{result['answer']}")
            print(f"\nRelevance Score: {result['relevance_score']}")
            print(f"Quality: {result['metadata']['evaluation']['quality']}")
            print(f"Chunks used: {result['metadata']['chunks_used']}")
            return True
        else:
            print(f"❌ Question failed: {response.status_code}")
            print(response.text)
            return False
    
    except Exception as e:
        print(f"❌ Question error: {str(e)}")
        return False

def test_summary():
    """Test document summary endpoint"""
    print("\n" + "="*60)
    print("4. Testing Document Summary...")
    print("="*60)
    
    try:
        response = requests.get(f"{BASE_URL}/summary")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Summary retrieved!")
            print(f"\nDocument: {result['document_name']}")
            print(f"Total chunks: {result['total_chunks']}")
            print(f"\nSummary:\n{result['summary']}")
            return True
        else:
            print(f"❌ Summary failed: {response.status_code}")
            print(response.text)
            return False
    
    except Exception as e:
        print(f"❌ Summary error: {str(e)}")
        return False

def test_stats():
    """Test stats endpoint"""
    print("\n" + "="*60)
    print("5. Testing System Stats...")
    print("="*60)
    
    try:
        response = requests.get(f"{BASE_URL}/stats")
        
        if response.status_code == 200:
            print("✅ Stats retrieved!")
            print(json.dumps(response.json(), indent=2))
            return True
        else:
            print(f"❌ Stats failed: {response.status_code}")
            return False
    
    except Exception as e:
        print(f"❌ Stats error: {str(e)}")
        return False

def run_full_test(file_path: str, questions: list):
    """Run complete test suite"""
    print("\n" + "="*70)
    print("🧪 AI KNOWLEDGE HELPER - API TEST SUITE")
    print("="*70)
    
    results = []
    
    # Test 1: Health check
    results.append(test_health_check())
    
    if not results[0]:
        print("\n❌ Server not running. Exiting tests.")
        return
    
    # Test 2: Upload document
    results.append(test_upload(file_path))
    
    if not results[1]:
        print("\n❌ Upload failed. Cannot proceed with other tests.")
        return
    
    # Test 3: Ask questions
    for question in questions:
        results.append(test_ask_question(question))
    
    # Test 4: Get summary
    results.append(test_summary())
    
    # Test 5: Get stats
    results.append(test_stats())
    
    # Summary
    print("\n" + "="*70)
    print("📊 TEST RESULTS SUMMARY")
    print("="*70)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("\n🎉 All tests passed! Your RAG system is working perfectly!")
    else:
        print("\n⚠️ Some tests failed. Check the errors above.")
    
    print("="*70)

# ========== MAIN ==========
if __name__ == "__main__":
    # Default test configuration
    test_file = "sample.txt"  # Change this to your test file
    test_questions = [
        "What is the main topic of this document?",
        "Can you summarize the key points?",
        "What are the most important findings?"
    ]
    
    # Check if file path provided as argument
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    
    # Create a sample test file if it doesn't exist
    if not Path(test_file).exists():
        print(f"\n⚠️ Test file '{test_file}' not found.")
        print("Creating a sample test file...")
        
        sample_content = """
        Artificial Intelligence and Machine Learning
        
        Artificial Intelligence (AI) has revolutionized modern technology. Machine learning, 
        a subset of AI, enables computers to learn from data without explicit programming.
        
        Key applications include:
        1. Natural Language Processing - Understanding and generating human language
        2. Computer Vision - Image and video analysis
        3. Predictive Analytics - Forecasting trends based on historical data
        
        The field continues to grow rapidly with new breakthroughs in deep learning,
        neural networks, and reinforcement learning techniques.
        """
        
        with open(test_file, 'w') as f:
            f.write(sample_content)
        
        print(f"✅ Created sample file: {test_file}")
    
    # Run tests
    run_full_test(test_file, test_questions)
