#!/usr/bin/env python3
"""
Test script for Bark Text-to-Speech by Suno AI.
Bark is known for high-quality speech synthesis with support for multiple voices,
non-speech sounds, and natural prosody.
"""

import os
import torch
from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
from datetime import datetime

# Create output directory
output_dir = "manual_tests/output/bark_tts_test"
os.makedirs(output_dir, exist_ok=True)

# Available speaker prompts
SPEAKERS = {
    "announcer": "[announcer] This is a broadcast message",
    "professional": "[professional] Speaking in a clear, articulate manner",
    "excited": "[excited] Wow! This is amazing news!",
    "casual": "[casual] Just having a relaxed conversation",
    "authoritative": "[authoritative] Listen carefully to these instructions"
}

# Test text samples
TEXTS = {
    "statement": "The quick brown fox jumps over the lazy dog.",
    "question": "How does changing voices affect the speech quality?",
    "emphasis": "This is VERY important and needs special attention!",
    "emotion": "I can't believe how amazing this discovery is!",
    "complex": """
    Let me tell you something interesting! *laughs* 
    AI has come so far in recent years. ♪ It's truly remarkable! ♪
    *sighs* But we still have so much to learn... 
    What do you think about that? *laughs*
    """
}

def save_audio(text, speaker_prompt, filename):
    """Generate speech using Bark and save to file."""
    print(f"Generating speech with prompt '{speaker_prompt}': {filename}")
    
    try:
        # Combine speaker prompt with text
        full_text = f"{speaker_prompt} {text}"
        
        # Generate audio
        audio_array = generate_audio(full_text)
        
        # Save to file
        filepath = f"{output_dir}/{filename}"
        write_wav(filepath, SAMPLE_RATE, audio_array)
        print(f"Saved to {filepath}")
        return True
        
    except Exception as e:
        print(f"Error generating speech: {str(e)}")
        return False

def test_voices():
    """Test all speaker prompts with different text types."""
    print("\n=== Testing Bark Voices ===")
    
    for speaker_name, prompt in SPEAKERS.items():
        for text_type, text in TEXTS.items():
            filename = f"{speaker_name}_{text_type}.wav"
            save_audio(text, prompt, filename)

def test_long_form():
    """Test with longer text to demonstrate natural prosody."""
    print("\n=== Testing Long-form Text ===")
    
    long_text = """
    Artificial intelligence has transformed the way we interact with technology.
    *pauses thoughtfully* 
    From virtual assistants to autonomous vehicles, AI systems are becoming 
    increasingly sophisticated. ♪ It's quite remarkable! ♪
    
    However, with great power comes great responsibility.
    *serious tone* We must ensure that AI development remains ethical and 
    beneficial to humanity.
    
    What do you think about the future of AI? *curious tone*
    How will it change our lives in the next decade?
    """
    
    # Test with professional and authoritative voices
    for speaker_name in ["professional", "authoritative"]:
        filename = f"{speaker_name}_long_form.wav"
        save_audio(long_text, SPEAKERS[speaker_name], filename)

def test_multilingual():
    """Test multilingual capabilities."""
    print("\n=== Testing Multilingual Support ===")
    
    multilingual_texts = {
        "english": "Hello! How are you today?",
        "spanish": "¡Hola! ¿Cómo estás hoy?",
        "french": "Bonjour! Comment allez-vous aujourd'hui?",
        "german": "Hallo! Wie geht es Ihnen heute?"
    }
    
    for lang, text in multilingual_texts.items():
        filename = f"multilingual_{lang}.wav"
        save_audio(text, SPEAKERS["professional"], filename)

def main():
    """Run all Bark TTS tests."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Starting Bark TTS tests at {timestamp}")
    print(f"Output directory: {output_dir}")
    
    # Preload models (this may take a while the first time)
    print("Preloading Bark models...")
    preload_models()
    
    test_voices()
    test_long_form()
    test_multilingual()
    
    print("\nAll tests completed. Check the output directory for results.")
    print("\nFeatures demonstrated:")
    print("- Multiple voice styles through speaker prompts")
    print("- Non-speech sounds (laughs, sighs, pauses)")
    print("- Musical tones (♪)")
    print("- Multilingual support")
    print("- Natural prosody and emphasis")

if __name__ == "__main__":
    main() 