#!/bin/bash
# Script to run all TTS tests with different voices and formats

# Check if the TTS service is running
if ! curl -s http://localhost:8000/health > /dev/null; then
    echo "Error: TTS service is not running. Please start it with 'make run-local' first."
    exit 1
fi

# Create output directories for different formats
mkdir -p output/wav
mkdir -p output/mp3
mkdir -p output/ogg

# Get a list of available voices
VOICES=$(curl -s http://localhost:8000/voices | python3 -c "import sys, json; print(' '.join(json.load(sys.stdin)['voices'][:5]))")

# If no voices are available, use a default one
if [ -z "$VOICES" ]; then
    VOICES="p225"
fi

# Select the first 3 voices (to avoid too many tests)
SELECTED_VOICES=$(echo $VOICES | tr ' ' '\n' | head -3 | tr '\n' ' ')

echo "Running tests with voices: $SELECTED_VOICES"

# Run tests with different voices and formats
for VOICE in $SELECTED_VOICES; do
    echo "Testing with voice: $VOICE"
    
    # WAV format
    echo "  - WAV format"
    python3 test_tts.py --voice "$VOICE" --format wav --output-dir output/wav
    
    # MP3 format
    echo "  - MP3 format"
    python3 test_tts.py --voice "$VOICE" --format mp3 --output-dir output/mp3
    
    # OGG format
    echo "  - OGG format"
    python3 test_tts.py --voice "$VOICE" --format ogg --output-dir output/ogg
    
    # Different speeds
    echo "  - Different speeds (WAV format)"
    python3 test_tts.py --voice "$VOICE" --format wav --speed 0.8 --output-dir output/wav --input-dir input
    python3 test_tts.py --voice "$VOICE" --format wav --speed 1.2 --output-dir output/wav --input-dir input
done

echo "All tests completed. Output files are in the 'output' directory."
echo "You can play the audio files with:"
echo "  - WAV: afplay output/wav/sample1.wav"
echo "  - MP3: afplay output/mp3/sample1.mp3"
echo "  - OGG: afplay output/ogg/sample1.ogg" 