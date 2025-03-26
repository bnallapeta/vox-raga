#!/usr/bin/env python3
"""
Advanced test script for Bark Text-to-Speech that demonstrates various features
including emotions, laughter, non-speech sounds, and speaker prompts.

Features demonstrated:
- Different speaker prompts
- Emotional expressions
- Non-speech sounds (laughter, sighs, etc.)
- Musical notes
- Background sounds
- Multi-speaker conversations
"""

import os
os.environ['TORCH_LOAD_WEIGHTS_ONLY'] = '0'  # Disable weights_only for torch.load

from bark import SAMPLE_RATE, generate_audio, preload_models
import scipy.io.wavfile as wavfile
from datetime import datetime
import time

# Create output directory
output_dir = "manual_tests/output/bark_advanced_test"
os.makedirs(output_dir, exist_ok=True)

# Speaker prompts from Bark's examples
SPEAKER_PROMPTS = {
    "announcer": "[announcer] Breaking news!",
    "narrator": "[narrator] Let me tell you a story...",
    "old_man": "[old man] Back in my day...",
    "young_woman": "[young woman] Oh my gosh!",
    "young_man": "[young man] Hey, what's up?",
    "old_woman": "[old woman] Let me share some wisdom...",
    "excited_girl": "[excited girl] This is so amazing!",
    "stern_teacher": "[stern teacher] Pay attention, class.",
    "british_man": "[british man] Jolly good show!",
    "radio_host": "[radio host] And now for the weather..."
}

# Emotion prompts with special effects
EMOTION_PROMPTS = {
    "happy": {
        "prompt": "♪ This makes me so happy! *laughs* What a wonderful day! ♪",
        "desc": "Happy with singing and laughter"
    },
    "sad": {
        "prompt": "*sighs deeply* This is so sad... *sniffs* I can't believe it...",
        "desc": "Sad with sighs and sniffling"
    },
    "angry": {
        "prompt": "WHAT?! *furious breathing* This is UNACCEPTABLE! *slams table*",
        "desc": "Angry with dramatic effects"
    },
    "surprised": {
        "prompt": "*gasps* OH. MY. GOODNESS! *shocked pause* I can't believe it!",
        "desc": "Surprised with gasps"
    },
    "thoughtful": {
        "prompt": "Hmm... *contemplative pause* That's quite interesting... *soft chuckle*",
        "desc": "Thoughtful with pauses"
    }
}

# Conversation scenarios
CONVERSATIONS = {
    "cafe_scene": [
        ("[young_woman] Hi! Can I get a coffee? *cheerful*",
         "casual_order_1"),
        ("[young_man] Sure! *friendly* Would you like it hot or iced? ♪",
         "casual_response_1"),
        ("[young_woman] Iced please! *excited* It's such a warm day! *laughs*",
         "casual_order_2")
    ],
    "news_scene": [
        ("[announcer] Breaking news! *dramatic pause*",
         "news_intro"),
        ("[reporter] We're live at the scene *background crowd noise*",
         "news_report"),
        ("[witness] *nervous* I saw the whole thing! *excited*",
         "news_witness")
    ]
}

def save_audio(text, filename, speaker_prompt=""):
    """Generate and save audio using Bark."""
    print(f"\nGenerating: {filename}")
    print(f"Text: {text}")
    
    try:
        # Combine speaker prompt with text if provided
        full_text = f"{speaker_prompt} {text}" if speaker_prompt else text
        
        # Generate audio
        audio_array = generate_audio(full_text)
        
        # Save to file
        filepath = f"{output_dir}/{filename}.wav"
        wavfile.write(filepath, SAMPLE_RATE, audio_array)
        print(f"✓ Saved to {filepath}")
        
        # Add a small delay to avoid overwhelming the system
        time.sleep(1)
        return True
    
    except Exception as e:
        print(f"✗ Error generating audio: {str(e)}")
        return False

def test_speaker_variations():
    """Test different speaker prompts."""
    print("\n=== Testing Speaker Variations ===")
    
    base_text = "The quick brown fox jumps over the lazy dog."
    for speaker, prompt in SPEAKER_PROMPTS.items():
        save_audio(base_text, f"speaker_{speaker}", prompt)

def test_emotions():
    """Test emotional expressions with special effects."""
    print("\n=== Testing Emotions and Effects ===")
    
    for emotion, data in EMOTION_PROMPTS.items():
        save_audio(data["prompt"], f"emotion_{emotion}")

def test_conversations():
    """Test multi-speaker conversations with effects."""
    print("\n=== Testing Conversations ===")
    
    for scene, lines in CONVERSATIONS.items():
        print(f"\nGenerating scene: {scene}")
        for text, filename in lines:
            save_audio(text, f"{scene}_{filename}")

def test_special_effects():
    """Test various special effects and non-speech sounds."""
    print("\n=== Testing Special Effects ===")
    
    effects = [
        ("*laughs* This is hilarious! *wipes tear* Oh my goodness! *continues laughing*",
         "effect_laughter"),
        ("♪ La la la! ♪ This is such a wonderful song! ♪ *dances* ♪",
         "effect_singing"),
        ("*yawns* I'm so tired... *stretches* Time for bed... *soft snore*",
         "effect_sleepy"),
        ("*whispers* Don't make a sound... *footsteps* *dramatic pause* *gasps*",
         "effect_suspense"),
        ("*clears throat* ATTENTION EVERYONE! *taps microphone* *feedback noise*",
         "effect_announcement")
    ]
    
    for text, filename in effects:
        save_audio(text, filename)

def main():
    """Run all Bark advanced tests."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Starting Bark advanced tests at {timestamp}")
    print(f"Output directory: {output_dir}")
    
    # Preload models
    print("\nPreloading Bark models (this may take a while)...")
    preload_models()
    
    # Run tests
    test_speaker_variations()
    test_emotions()
    test_conversations()
    test_special_effects()
    
    print("\n=== All tests completed! ===")
    print("\nFeatures demonstrated:")
    print("- Multiple speaker voices through prompts")
    print("- Emotional expressions with special effects")
    print("- Non-speech sounds (laughs, gasps, sighs)")
    print("- Musical elements (♪)")
    print("- Multi-speaker conversations")
    print("- Background effects and ambiance")
    print(f"\nCheck the output directory: {output_dir}")

if __name__ == "__main__":
    main() 