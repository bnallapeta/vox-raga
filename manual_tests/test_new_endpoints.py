#!/usr/bin/env python3
"""
Test script for the new TTS API endpoints.
"""
import requests
import json
import time
import os
import asyncio
import zipfile
import io

try:
    import websocket
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False

# Base URL for the API
BASE_URL = "http://localhost:8000"

def test_batch_synthesis():
    """Test batch synthesis endpoint."""
    print("\n=== Testing Batch Synthesis Endpoint ===")
    
    # Request data
    data = {
        "texts": [
            "Hello, this is the first test sentence.",
            "This is the second test sentence.",
            "And this is the third test sentence."
        ],
        "options": {
            "language": "en",
            "voice": "p225",
            "speed": 1.0,
            "format": "wav"
        }
    }
    
    # Make the request
    response = requests.post(f"{BASE_URL}/batch_synthesize", json=data)
    
    # Check the response
    if response.status_code == 200:
        print("Batch synthesis successful!")
        print(f"Content type: {response.headers['content-type']}")
        print(f"Processing time: {response.headers.get('X-Processing-Time', 'N/A')} seconds")
        
        # Save the ZIP file
        with open("batch_synthesis.zip", "wb") as f:
            f.write(response.content)
        
        # Extract the ZIP file
        with zipfile.ZipFile("batch_synthesis.zip", "r") as zip_file:
            # Print the contents
            print(f"ZIP file contains {len(zip_file.namelist())} files:")
            for filename in zip_file.namelist():
                print(f"  - {filename}")
            
            # Extract the files
            zip_file.extractall("batch_output")
        
        print("Files extracted to 'batch_output' directory")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)


def test_async_synthesis():
    """Test async synthesis endpoint."""
    print("\n=== Testing Async Synthesis Endpoint ===")
    
    # Request data
    data = {
        "text": "This is a test of the asynchronous synthesis endpoint.",
        "options": {
            "language": "en",
            "voice": "p225",
            "speed": 1.0,
            "format": "wav"
        }
    }
    
    # Submit the job
    response = requests.post(f"{BASE_URL}/synthesize/async", json=data)
    
    # Check the response
    if response.status_code == 202:
        print("Async synthesis job submitted successfully!")
        job_id = response.json()["job_id"]
        print(f"Job ID: {job_id}")
        
        # Poll for job completion
        print("Polling for job completion...")
        while True:
            status_response = requests.get(f"{BASE_URL}/synthesize/status/{job_id}")
            status = status_response.json()["status"]
            print(f"Job status: {status}")
            
            if status == "completed":
                break
            elif status == "failed":
                print(f"Job failed: {status_response.json().get('error', 'Unknown error')}")
                return
            
            time.sleep(1)
        
        # Get the result
        result_response = requests.get(f"{BASE_URL}/synthesize/result/{job_id}")
        
        # Check the result
        if result_response.status_code == 200:
            print("Async synthesis result retrieved successfully!")
            print(f"Content type: {result_response.headers['content-type']}")
            
            # Save the audio file
            with open("async_synthesis.wav", "wb") as f:
                f.write(result_response.content)
            
            print("Audio saved to 'async_synthesis.wav'")
        else:
            print(f"Error retrieving result: {result_response.status_code}")
            print(result_response.text)
    else:
        print(f"Error submitting job: {response.status_code}")
        print(response.text)


def test_voices_by_language():
    """Test voices by language endpoint."""
    print("\n=== Testing Voices by Language Endpoint ===")
    
    # List of languages to test
    languages = ["en", "fr", "de", "es", "it"]
    
    for language in languages:
        # Make the request
        response = requests.get(f"{BASE_URL}/voices/{language}")
        
        # Check the response
        if response.status_code == 200:
            voices = response.json()["voices"]
            print(f"Language: {language}, Voices: {len(voices)}")
            if voices:
                print(f"  Sample voices: {voices[:3]}")
        else:
            print(f"Error for language {language}: {response.status_code}")
            print(response.text)


def test_simple_synthesis():
    """Test simple synthesis without emotion or style."""
    print("\n=== Testing Simple Synthesis ===")
    
    # Request data
    data = {
        "text": "This is a test of the simple synthesis endpoint.",
        "options": {
            "language": "en",
            "voice": "p225",
            "speed": 1.0,
            "format": "wav"
        }
    }
    
    # Make the request
    response = requests.post(f"{BASE_URL}/synthesize", json=data)
    
    # Check the response
    if response.status_code == 200:
        print("Simple synthesis successful!")
        print(f"Content type: {response.headers['content-type']}")
        print(f"Processing time: {response.headers.get('X-Processing-Time', 'N/A')} seconds")
        
        # Save the audio file
        with open("simple_synthesis.wav", "wb") as f:
            f.write(response.content)
        
        print("Audio saved to 'simple_synthesis.wav'")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)


def test_default_speaker():
    """Test synthesis with default speaker."""
    print("\n=== Testing Default Speaker Behavior ===")
    
    # Request data
    data = {
        "text": "This is a test of the default speaker behavior.",
        "options": {
            "language": "en",
            "voice": "default",
            "speed": 1.0,
            "format": "wav"
        }
    }
    
    # Make the request
    response = requests.post(f"{BASE_URL}/synthesize", json=data)
    
    # Check the response
    if response.status_code == 200:
        print("Default speaker synthesis successful!")
        print(f"Content type: {response.headers['content-type']}")
        print(f"Processing time: {response.headers.get('X-Processing-Time', 'N/A')} seconds")
        
        # Save the audio file
        with open("default_speaker_synthesis.wav", "wb") as f:
            f.write(response.content)
        
        print("Audio saved to 'default_speaker_synthesis.wav'")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)


def test_websocket():
    """Test WebSocket endpoint."""
    print("\n=== Testing WebSocket Endpoint ===")
    
    # Create a WebSocket connection
    ws = websocket.WebSocket()
    ws.connect(f"ws://localhost:8000/synthesize/ws")
    
    # Request data - simple text without emotion or style
    data = {
        "text": "This is a test of the WebSocket synthesis endpoint.",
        "options": {
            "language": "en",
            "voice": "p225",
            "speed": 1.0,
            "format": "wav"
        }
    }
    
    # Send the request
    ws.send(json.dumps(data))
    
    # Receive the response
    print("Waiting for response...")
    response = ws.recv()
    
    # Check if the response is binary
    if isinstance(response, bytes):
        print("WebSocket synthesis successful!")
        
        # Save the audio file
        with open("websocket_synthesis.wav", "wb") as f:
            f.write(response)
        
        print("Audio saved to 'websocket_synthesis.wav'")
    else:
        print(f"Error: {response}")
    
    # Close the connection
    ws.close()


def main():
    """Run all tests."""
    # Create output directory
    os.makedirs("batch_output", exist_ok=True)
    
    # Run tests
    test_batch_synthesis()
    test_async_synthesis()
    test_voices_by_language()
    test_simple_synthesis()
    test_default_speaker()
    
    # WebSocket test requires the websocket-client package
    if WEBSOCKET_AVAILABLE:
        test_websocket()
    else:
        print("\nSkipping WebSocket test - websocket-client package not installed")
        print("Install with: pip install websocket-client")


if __name__ == "__main__":
    main() 