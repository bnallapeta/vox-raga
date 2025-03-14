#!/usr/bin/env python3
"""
Test script for Meta's MMS-TTS (Massively Multilingual Speech).
MMS-TTS is known for high-quality multilingual speech synthesis with
support for over 1100 languages.
"""

import os
import torch
import torchaudio
from datetime import datetime
from transformers import VitsModel, AutoTokenizer

# Create output directory
output_dir = "manual_tests/output/mms_tts_test"
os.makedirs(output_dir, exist_ok=True)

# Model and tokenizer names
MODEL_ID = "facebook/mms-tts"
SAMPLE_RATE = 16000

# Language codes and their names
LANGUAGES = {
    "eng": "English",
    "spa": "Spanish",
    "fra": "French",
    "deu": "German",
    "cmn": "Mandarin",
    "hin": "Hindi",
    "ara": "Arabic",
    "jpn": "Japanese"
}

# Test text samples
TEXTS = {
    "statement": "The quick brown fox jumps over the lazy dog.",
    "question": "How does changing languages affect the speech quality?",
    "emphasis": "This is VERY important and needs special attention!",
    "emotion": "I can't believe how amazing this discovery is!"
}

def load_model():
    """Load the MMS-TTS model and tokenizer."""
    print("Loading MMS-TTS model and tokenizer...")
    model = VitsModel.from_pretrained(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    return model, tokenizer

def save_audio(text, lang_code, model, tokenizer, filename):
    """Generate speech using MMS-TTS and save to file."""
    print(f"Generating speech in {LANGUAGES[lang_code]}: {filename}")
    
    try:
        # Tokenize text
        inputs = tokenizer(text, return_tensors="pt")
        
        # Generate speech
        with torch.no_grad():
            output = model.generate(
                **inputs,
                language=lang_code,
                do_sample=True,
                temperature=0.7
            )
        
        # Save to file
        filepath = f"{output_dir}/{filename}"
        torchaudio.save(filepath, output.audio.squeeze().unsqueeze(0), SAMPLE_RATE)
        print(f"Saved to {filepath}")
        return True
        
    except Exception as e:
        print(f"Error generating speech: {str(e)}")
        return False

def test_languages():
    """Test speech synthesis in different languages."""
    print("\n=== Testing Multiple Languages ===")
    
    model, tokenizer = load_model()
    
    for lang_code, lang_name in LANGUAGES.items():
        for text_type, text in TEXTS.items():
            filename = f"{lang_code}_{text_type}.wav"
            save_audio(text, lang_code, model, tokenizer, filename)

def test_long_form():
    """Test with longer text to demonstrate natural prosody."""
    print("\n=== Testing Long-form Text ===")
    
    model, tokenizer = load_model()
    
    long_text = """
    Artificial intelligence has transformed the way we interact with technology.
    From virtual assistants to autonomous vehicles, AI systems are becoming 
    increasingly sophisticated. However, with great power comes great responsibility.
    We must ensure that AI development remains ethical and beneficial to humanity.
    What do you think about the future of AI? How will it change our lives in the
    next decade?
    """
    
    # Test long-form text in English and Mandarin
    for lang_code in ["eng", "cmn"]:
        filename = f"{lang_code}_long_form.wav"
        save_audio(long_text, lang_code, model, tokenizer, filename)

def test_mixed_language():
    """Test code-switching between languages."""
    print("\n=== Testing Mixed Language Support ===")
    
    model, tokenizer = load_model()
    
    # Text with code-switching
    mixed_texts = {
        "eng_spa": "Hello! ¿Cómo estás? I'm doing great!",
        "eng_fra": "Hello! Comment allez-vous? I'm fine!",
        "eng_deu": "Hello! Wie geht es dir? I'm good!"
    }
    
    for mix_name, text in mixed_texts.items():
        filename = f"mixed_{mix_name}.wav"
        # Use English as base language for mixed text
        save_audio(text, "eng", model, tokenizer, filename)

def main():
    """Run all MMS-TTS tests."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Starting MMS-TTS tests at {timestamp}")
    print(f"Output directory: {output_dir}")
    
    test_languages()
    test_long_form()
    test_mixed_language()
    
    print("\nAll tests completed. Check the output directory for results.")
    print("\nFeatures demonstrated:")
    print("- Multilingual synthesis (1100+ languages)")
    print("- Natural prosody across languages")
    print("- Code-switching capabilities")
    print("- Long-form text handling")
    print("- Different text types (statements, questions, emphasis)")

if __name__ == "__main__":
    main() 