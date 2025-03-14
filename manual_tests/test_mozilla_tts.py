#!/usr/bin/env python3
"""
Test script for Mozilla TTS.
Mozilla TTS is an open source Text-to-Speech engine, with various models and features
including multi-speaker support and custom voice training capabilities.
"""

import os
import torch
from datetime import datetime
from TTS.api import TTS

# Create output directory
output_dir = "manual_tests/output/mozilla_tts_test"
os.makedirs(output_dir, exist_ok=True)

# Available models and voices
MODELS = {
    "tts_models/en/ljspeech/tacotron2-DDC": "LJSpeech - Female voice",
    "tts_models/en/vctk/vits": "VCTK - Multi-speaker",
    "tts_models/multilingual/multi-dataset/xtts_v2": "XTTS v2 - Multilingual"
}

# Test text samples
TEXTS = {
    "statement": "The quick brown fox jumps over the lazy dog.",
    "question": "How does changing voices affect the speech quality?",
    "emphasis": "This is VERY important and needs special attention!",
    "emotion": "I can't believe how amazing this discovery is!"
}

def load_tts(model_name):
    """Load a TTS model."""
    print(f"Loading model: {model_name}")
    try:
        # Initialize TTS with the selected model
        tts = TTS(model_name=model_name, progress_bar=True)
        return tts
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def save_audio(text, tts, speaker=None, language=None, filename=None):
    """Generate speech using Mozilla TTS and save to file."""
    print(f"Generating speech with model '{tts.model_name}': {filename}")
    
    try:
        # Prepare kwargs based on model capabilities
        kwargs = {}
        if speaker and hasattr(tts, "speakers") and speaker in tts.speakers:
            kwargs["speaker"] = speaker
        if language and hasattr(tts, "languages") and language in tts.languages:
            kwargs["language"] = language
        
        # Generate audio
        filepath = f"{output_dir}/{filename}"
        tts.tts_to_file(text=text, file_path=filepath, **kwargs)
        print(f"Saved to {filepath}")
        return True
        
    except Exception as e:
        print(f"Error generating speech: {str(e)}")
        return False

def test_models():
    """Test different TTS models."""
    print("\n=== Testing Different Models ===")
    
    for model_name, description in MODELS.items():
        print(f"\nTesting {description}")
        tts = load_tts(model_name)
        if tts is None:
            continue
        
        for text_type, text in TEXTS.items():
            filename = f"{model_name.split('/')[-1]}_{text_type}.wav"
            save_audio(text, tts, filename=filename)

def test_vctk_voices():
    """Test multiple voices using the VCTK model."""
    print("\n=== Testing VCTK Voices ===")
    
    tts = load_tts("tts_models/en/vctk/vits")
    if tts is None:
        return
    
    # Test first few speakers
    test_speakers = list(tts.speakers)[:5] if hasattr(tts, "speakers") else []
    
    for speaker in test_speakers:
        for text_type, text in TEXTS.items():
            filename = f"vctk_{speaker}_{text_type}.wav"
            save_audio(text, tts, speaker=speaker, filename=filename)

def test_multilingual():
    """Test multilingual capabilities using XTTS v2."""
    print("\n=== Testing Multilingual Support ===")
    
    tts = load_tts("tts_models/multilingual/multi-dataset/xtts_v2")
    if tts is None:
        return
    
    multilingual_texts = {
        "en": "Hello! How are you today?",
        "es": "¡Hola! ¿Cómo estás hoy?",
        "fr": "Bonjour! Comment allez-vous aujourd'hui?",
        "de": "Hallo! Wie geht es Ihnen heute?"
    }
    
    for lang, text in multilingual_texts.items():
        filename = f"xtts_multilingual_{lang}.wav"
        save_audio(text, tts, language=lang, filename=filename)

def test_long_form():
    """Test with longer text to demonstrate natural prosody."""
    print("\n=== Testing Long-form Text ===")
    
    tts = load_tts("tts_models/en/ljspeech/tacotron2-DDC")
    if tts is None:
        return
    
    long_text = """
    Artificial intelligence has transformed the way we interact with technology.
    From virtual assistants to autonomous vehicles, AI systems are becoming 
    increasingly sophisticated. However, with great power comes great responsibility.
    We must ensure that AI development remains ethical and beneficial to humanity.
    What do you think about the future of AI? How will it change our lives in the
    next decade?
    """
    
    filename = "long_form_tacotron2.wav"
    save_audio(long_text, tts, filename=filename)

def main():
    """Run all Mozilla TTS tests."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Starting Mozilla TTS tests at {timestamp}")
    print(f"Output directory: {output_dir}")
    
    test_models()
    test_vctk_voices()
    test_multilingual()
    test_long_form()
    
    print("\nAll tests completed. Check the output directory for results.")
    print("\nFeatures demonstrated:")
    print("- Multiple TTS models (Tacotron2, VITS, XTTS)")
    print("- Multi-speaker synthesis with VCTK")
    print("- Multilingual support with XTTS v2")
    print("- Long-form text handling")
    print("- Different text types (statements, questions, emphasis)")

if __name__ == "__main__":
    main() 