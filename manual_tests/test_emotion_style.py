#!/usr/bin/env python3
"""
Manual test script for testing emotion and style parameters in the VoxRaga API.
This script sends requests to the API with different emotion and style parameters
and saves the resulting audio files for comparison.
"""

import os
import requests
import time
from datetime import datetime

# API endpoint
API_URL = "http://localhost:8000/synthesize"

# Create output directory
output_dir = "manual_tests/output/emotion_style_test"
os.makedirs(output_dir, exist_ok=True)

# Test text
text = "The quick brown fox jumps over the lazy dog."

# Test parameters
emotions = ["happy", "sad", "angry", "neutral", "excited", "calm"]
styles = ["formal", "casual", "news", "storytelling", "conversational", "instructional"]

# Default options
default_options = {
    "language": "en",
    "voice": "p225",
    "speed": 1.0,
    "format": "wav"
}

def test_emotion():
    """Test different emotion parameters."""
    print("\n=== Testing Emotion Parameters ===")
    
    # First, generate a baseline with no emotion
    baseline_options = default_options.copy()
    baseline_file = f"{output_dir}/baseline.wav"
    
    print(f"Generating baseline with no emotion...")
    response = requests.post(
        API_URL,
        json={"text": text, "options": baseline_options}
    )
    
    if response.status_code == 200:
        with open(baseline_file, "wb") as f:
            f.write(response.content)
        print(f"Saved baseline to {baseline_file}")
    else:
        print(f"Error generating baseline: {response.status_code}")
        print(response.text)
    
    # Test each emotion
    for emotion in emotions:
        options = default_options.copy()
        options["emotion"] = emotion
        output_file = f"{output_dir}/emotion_{emotion}.wav"
        
        print(f"Testing emotion: {emotion}...")
        response = requests.post(
            API_URL,
            json={"text": text, "options": options}
        )
        
        if response.status_code == 200:
            with open(output_file, "wb") as f:
                f.write(response.content)
            print(f"Saved {emotion} to {output_file}")
        else:
            print(f"Error with emotion {emotion}: {response.status_code}")
            print(response.text)
        
        # Add a small delay to avoid overwhelming the server
        time.sleep(1)

def test_style():
    """Test different style parameters."""
    print("\n=== Testing Style Parameters ===")
    
    # Test each style
    for style in styles:
        options = default_options.copy()
        options["style"] = style
        output_file = f"{output_dir}/style_{style}.wav"
        
        print(f"Testing style: {style}...")
        response = requests.post(
            API_URL,
            json={"text": text, "options": options}
        )
        
        if response.status_code == 200:
            with open(output_file, "wb") as f:
                f.write(response.content)
            print(f"Saved {style} to {output_file}")
        else:
            print(f"Error with style {style}: {response.status_code}")
            print(response.text)
        
        # Add a small delay to avoid overwhelming the server
        time.sleep(1)

def test_emotion_style_combination():
    """Test combinations of emotion and style parameters."""
    print("\n=== Testing Emotion and Style Combinations ===")
    
    # Test a few combinations
    combinations = [
        ("happy", "conversational"),
        ("sad", "formal"),
        ("excited", "storytelling"),
        ("calm", "instructional")
    ]
    
    for emotion, style in combinations:
        options = default_options.copy()
        options["emotion"] = emotion
        options["style"] = style
        output_file = f"{output_dir}/combo_{emotion}_{style}.wav"
        
        print(f"Testing combination: {emotion} + {style}...")
        response = requests.post(
            API_URL,
            json={"text": text, "options": options}
        )
        
        if response.status_code == 200:
            with open(output_file, "wb") as f:
                f.write(response.content)
            print(f"Saved {emotion}+{style} to {output_file}")
        else:
            print(f"Error with {emotion}+{style}: {response.status_code}")
            print(response.text)
        
        # Add a small delay to avoid overwhelming the server
        time.sleep(1)

def main():
    """Run all tests."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Starting emotion and style tests at {timestamp}")
    print(f"Output directory: {output_dir}")
    
    test_emotion()
    test_style()
    test_emotion_style_combination()
    
    print("\nAll tests completed. Check the output directory for results.")

if __name__ == "__main__":
    main() 