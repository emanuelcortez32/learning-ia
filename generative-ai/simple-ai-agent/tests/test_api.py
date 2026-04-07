"""
Test client for the AI Agent API
Usage:
    # Test regular endpoint
    python test_api.py

    # Test streaming endpoint
    python test_api.py --stream
"""
import requests
import json
import sys

BASE_URL = "http://localhost:8000"


def test_regular_chat():
    """Test the regular /chat endpoint"""
    print("Testing regular chat endpoint...\n")
    
    response = requests.post(
        f"{BASE_URL}/chat",
        json={"query": "I need a jacket for winter. What do you recommend?"}
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"Response: {data['response']}\n")
    else:
        print(f"Error: {response.status_code} - {response.text}\n")


def test_streaming_chat():
    """Test the streaming /chat/stream endpoint"""
    print("Testing streaming chat endpoint...\n")
    
    response = requests.post(
        f"{BASE_URL}/chat/stream",
        json={"query": "I need a jacket for winter. What do you recommend?"},
        stream=True
    )
    
    if response.status_code == 200:
        print("Streaming response: ", end="", flush=True)
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data_str = line[6:]  # Remove 'data: ' prefix
                    if data_str == '[DONE]':
                        print("\n\nStream completed!")
                        break
                    try:
                        data = json.loads(data_str)
                        print(data.get('content', ''), end="", flush=True)
                    except json.JSONDecodeError:
                        pass
    else:
        print(f"Error: {response.status_code} - {response.text}\n")


def test_health():
    """Test the health endpoint"""
    response = requests.get(f"{BASE_URL}/health")
    print(f"Health check: {response.json()}\n")


if __name__ == "__main__":
    # Check if API is running
    try:
        test_health()
    except requests.exceptions.ConnectionError:
        print("Error: API is not running. Please start the API first with:")
        print("  python src/api.py")
        sys.exit(1)
    
    # Run tests based on arguments
    if "--stream" in sys.argv:
        test_streaming_chat()
    else:
        test_regular_chat()
