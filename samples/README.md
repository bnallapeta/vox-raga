# VoxRaga Samples

This directory contains sample files and scripts for testing the VoxRaga Text-to-Speech (TTS) service.

## Directory Structure

- `input/`: Contains sample text files to be processed by the TTS service
- `output/`: Where generated audio files are stored
- `test_tts.py`: Python script for testing the TTS service
- `run_all_tests.sh`: Shell script to run tests with different voices and formats

## Sample Text Files

The `input/` directory contains the following sample text files:

- `sample1.txt`: A pangram (sentence containing all letters of the alphabet)
- `sample2.txt`: A passage about artificial intelligence
- `sample3.txt`: Text with questions and exclamations to test intonation

## Prerequisites

Before running the tests, make sure:

1. The VoxRaga TTS service is running on `http://localhost:8000`
2. You have Python 3.x installed with the `requests` module

## Quick Start

### List Available Voices and Languages

```bash
# List available voices
python test_tts.py --list-voices

# List available languages
python test_tts.py --list-languages
```

### Generate Audio Files

```bash
# Generate WAV files with default voice
python test_tts.py --format wav

# Try a different voice
python test_tts.py --voice p226 --format wav

# Adjust speech speed
python test_tts.py --voice p225 --speed 1.2 --format wav

# Generate MP3 files
python test_tts.py --format mp3
```

### Run All Tests

To test multiple voices and formats at once:

```bash
./run_all_tests.sh
```

## Playing Audio Files from Terminal

- **macOS**: `afplay output/sample1.wav`
- **Linux**: `aplay output/sample1.wav` or `mpg123 output/sample1.mp3`
- **Windows**: `start output/sample1.wav`

## For More Options

See the main README.md in the project root for detailed configuration options. 