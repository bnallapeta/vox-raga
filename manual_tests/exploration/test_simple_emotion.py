#!/usr/bin/env python3
"""
Simplified test script for testing emotion parameters in the VoxRaga API.
This script focuses only on 'excited' and 'sad' emotions for easier comparison.
"""

import os
import requests
import time
from datetime import datetime

# API endpoint
API_URL = "http://localhost:8000/synthesize"

# Create output directory
output_dir = "manual_tests/output/simple_emotion_test"
os.makedirs(output_dir, exist_ok=True)

# Test text
text = "The quick brown fox jumps over the lazy dog."

# Default options
default_options = {
    "language": "en",
    "voice": "p225",
    "speed": 1.0,
    "format": "wav"
}

def test_simple_emotions():
    """Test baseline, excited, and sad emotions."""
    print("\n=== Testing Simple Emotions ===")
    
    # Generate baseline with no emotion
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
    
    # Test excited emotion
    excited_options = default_options.copy()
    excited_options["emotion"] = "excited"
    excited_file = f"{output_dir}/excited.wav"
    
    print(f"Testing excited emotion...")
    response = requests.post(
        API_URL,
        json={"text": text, "options": excited_options}
    )
    
    if response.status_code == 200:
        with open(excited_file, "wb") as f:
            f.write(response.content)
        print(f"Saved excited to {excited_file}")
    else:
        print(f"Error with excited emotion: {response.status_code}")
        print(response.text)
    
    # Test sad emotion
    sad_options = default_options.copy()
    sad_options["emotion"] = "sad"
    sad_file = f"{output_dir}/sad.wav"
    
    print(f"Testing sad emotion...")
    response = requests.post(
        API_URL,
        json={"text": text, "options": sad_options}
    )
    
    if response.status_code == 200:
        with open(sad_file, "wb") as f:
            f.write(response.content)
        print(f"Saved sad to {sad_file}")
    else:
        print(f"Error with sad emotion: {response.status_code}")
        print(response.text)

def test_direct_speed_pitch():
    """Test direct speed and pitch adjustments without emotion labels."""
    print("\n=== Testing Direct Speed and Pitch Adjustments ===")
    
    # Test fast and high pitch (similar to excited)
    fast_high_options = default_options.copy()
    fast_high_options["speed"] = 1.2
    fast_high_options["pitch"] = 1.2
    fast_high_file = f"{output_dir}/fast_high.wav"
    
    print(f"Testing fast speed and high pitch...")
    response = requests.post(
        API_URL,
        json={"text": text, "options": fast_high_options}
    )
    
    if response.status_code == 200:
        with open(fast_high_file, "wb") as f:
            f.write(response.content)
        print(f"Saved fast_high to {fast_high_file}")
    else:
        print(f"Error with fast_high: {response.status_code}")
        print(response.text)
    
    # Test slow and low pitch (similar to sad)
    slow_low_options = default_options.copy()
    slow_low_options["speed"] = 0.8
    slow_low_options["pitch"] = 0.8
    slow_low_file = f"{output_dir}/slow_low.wav"
    
    print(f"Testing slow speed and low pitch...")
    response = requests.post(
        API_URL,
        json={"text": text, "options": slow_low_options}
    )
    
    if response.status_code == 200:
        with open(slow_low_file, "wb") as f:
            f.write(response.content)
        print(f"Saved slow_low to {slow_low_file}")
    else:
        print(f"Error with slow_low: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    print(f"Starting simple emotion test at {datetime.now().strftime('%Y%m%d_%H%M%S')}")
    print(f"Output directory: {output_dir}")
    
    test_simple_emotions()
    test_direct_speed_pitch()
    
    print("\nAll tests completed. Check the output directory for results.") 