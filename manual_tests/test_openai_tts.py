#!/usr/bin/env python3
"""
Test script for OpenAI's Text-to-Speech API.
This script demonstrates how to use OpenAI's TTS API with different voices and settings.
"""

import os
import time
from datetime import datetime
from openai import OpenAI

# Create output directory
output_dir = "manual_tests/output/openai_tts_test"
os.makedirs(output_dir, exist_ok=True)

# Initialize OpenAI client
client = OpenAI()  # Uses OPENAI_API_KEY environment variable

# Available voices in OpenAI's TTS
VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

# Test text samples
TEXTS = {
    "statement": "The quick brown fox jumps over the lazy dog.",
    "question": "How does changing voices affect the speech quality?",
    "emphasis": "This is VERY important and needs special attention!",
    "emotion": "I can't believe how amazing this discovery is!"
}

def save_speech(text, voice, filename):
    """Generate speech using OpenAI's TTS API and save to file."""
    print(f"Generating speech with voice '{voice}': {filename}")
    
    try:
        response = client.audio.speech.create(
            model="tts-1",  # or "tts-1-hd" for higher quality
            voice=voice,
            input=text
        )
        
        # Save to file
        filepath = f"{output_dir}/{filename}"
        response.stream_to_file(filepath)
        print(f"Saved to {filepath}")
        return True
        
    except Exception as e:
        print(f"Error generating speech: {str(e)}")
        return False
    
    # Add a small delay to avoid rate limits
    time.sleep(1)

def test_voices():
    """Test all available voices with different text types."""
    print("\n=== Testing OpenAI TTS Voices ===")
    
    for voice in VOICES:
        for text_type, text in TEXTS.items():
            filename = f"{voice}_{text_type}.mp3"
            save_speech(text, voice, filename)

def test_long_form():
    """Test with longer text to demonstrate natural prosody."""
    print("\n=== Testing Long-form Text ===")
    
    long_text = """
    Artificial intelligence has transformed the way we interact with technology.
    From virtual assistants to autonomous vehicles, AI systems are becoming 
    increasingly sophisticated. However, with great power comes great responsibility.
    We must ensure that AI development remains ethical and beneficial to humanity.
    What do you think about the future of AI? How will it change our lives in the
    next decade?
    """
    
    # Test with two contrasting voices
    for voice in ["nova", "onyx"]:  # Professional vs. Authoritative
        filename = f"{voice}_long_form.mp3"
        save_speech(long_text, voice, filename)

def main():
    """Run all OpenAI TTS tests."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Starting OpenAI TTS tests at {timestamp}")
    print(f"Output directory: {output_dir}")
    
    test_voices()
    test_long_form()
    
    print("\nAll tests completed. Check the output directory for results.")
    print("\nVoice characteristics:")
    print("- alloy: Versatile, neutral voice")
    print("- echo: Warm, conversational voice")
    print("- fable: Expressive, youthful voice")
    print("- onyx: Authoritative, deep voice")
    print("- nova: Professional, clear voice")
    print("- shimmer: Bright, energetic voice")

if __name__ == "__main__":
    main() 