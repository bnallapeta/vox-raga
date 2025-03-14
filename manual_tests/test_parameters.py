#!/usr/bin/env python3
"""
Test script for experimenting with direct speech parameter adjustments in the VoxRaga API.
Tests various combinations of pitch, speed, and energy parameters to find effective ranges.
"""

import os
import requests
import time
from datetime import datetime

# API endpoint
API_URL = "http://localhost:8000/synthesize"

# Create output directory
output_dir = "manual_tests/output/parameter_test"
os.makedirs(output_dir, exist_ok=True)

# Test text - using different texts to better hear the effects
TEXTS = {
    "statement": "The quick brown fox jumps over the lazy dog.",
    "question": "How does changing parameters affect the voice quality?",
    "emphasis": "This is VERY important and needs special attention!",
    "emotion": "I can't believe how amazing this discovery is!"
}

# Default options
default_options = {
    "language": "en",
    "voice": "p226",
    "format": "wav",
    "sample_rate": 22050
}

def save_audio(text, options, filename):
    """Helper function to send request and save audio file."""
    print(f"Generating: {filename}")
    response = requests.post(
        API_URL,
        json={"text": text, "options": options}
    )
    
    if response.status_code == 200:
        filepath = f"{output_dir}/{filename}"
        with open(filepath, "wb") as f:
            f.write(response.content)
        print(f"Saved to {filepath}")
        return True
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return False
    
def test_baseline():
    """Generate baseline audio files with default parameters."""
    print("\n=== Generating Baseline Samples ===")
    for text_type, text in TEXTS.items():
        options = default_options.copy()
        save_audio(text, options, f"baseline_{text_type}.wav")
        time.sleep(0.5)

def test_speed_variations():
    """Test different speech speeds."""
    print("\n=== Testing Speed Variations ===")
    speed_values = [0.5, 1.0, 2.0]
    
    for speed in speed_values:
        options = default_options.copy()
        options["speed"] = speed
        save_audio(
            TEXTS["statement"],
            options,
            f"speed_{int(speed*100)}.wav"
        )
        time.sleep(0.5)

def test_pitch_variations():
    """Test different pitch levels."""
    print("\n=== Testing Pitch Variations ===")
    
    # Test with both p226 and p227
    for speaker in ["p226", "p227"]:
        # Test more extreme pitch variations
        pitch_values = [0.5, 1.0, 2.0]
        
        for pitch in pitch_values:
            options = default_options.copy()
            options["voice"] = speaker
            options["pitch"] = pitch
            save_audio(
                TEXTS["statement"],
                options,
                f"pitch_{speaker}_{int(pitch*100)}.wav"
            )
            time.sleep(0.5)

def test_combinations():
    """Test combinations of parameters."""
    print("\n=== Testing Parameter Combinations ===")
    
    # Test cases designed to create distinct variations
    combinations = [
        # High pitch, fast speech
        {
            "pitch": 1.5,
            "speed": 1.3,
            "name": "high_fast"
        },
        # Low pitch, slow speech
        {
            "pitch": 0.7,
            "speed": 0.8,
            "name": "low_slow"
        },
        # Very high pitch
        {
            "pitch": 2.0,
            "speed": 1.0,
            "name": "very_high"
        },
        # Very low pitch
        {
            "pitch": 0.5,
            "speed": 1.0,
            "name": "very_low"
        }
    ]
    
    # Test each combination with both speakers
    for speaker in ["p226", "p227"]:
        for combo in combinations:
            options = default_options.copy()
            options["voice"] = speaker
            options["pitch"] = combo["pitch"]
            options["speed"] = combo["speed"]
            
            # Test with different text types
            for text_type, text in TEXTS.items():
                save_audio(
                    text,
                    options,
                    f"combo_{speaker}_{combo['name']}_{text_type}.wav"
                )
                time.sleep(0.5)

def main():
    """Run all parameter tests."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Starting parameter tests at {timestamp}")
    print(f"Output directory: {output_dir}")
    
    test_baseline()
    test_speed_variations()
    test_pitch_variations()
    test_combinations()
    
    print("\nAll tests completed. Check the output directory for results.")

if __name__ == "__main__":
    main() 