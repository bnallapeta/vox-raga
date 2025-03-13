#!/usr/bin/env python3
"""
Test script for the TTS service.
This script reads text from sample files and sends them to the TTS service for synthesis.
"""
import os
import sys
import json
import requests
import argparse
from pathlib import Path

def get_available_voices():
    """Get a list of available voices from the TTS service."""
    response = requests.get("http://localhost:8000/voices")
    if response.status_code == 200:
        return response.json()["voices"]
    else:
        print(f"Error getting voices: {response.status_code}")
        return []

def get_available_languages():
    """Get a list of available languages from the TTS service."""
    response = requests.get("http://localhost:8000/languages")
    if response.status_code == 200:
        return response.json()["languages"]
    else:
        print(f"Error getting languages: {response.status_code}")
        return []

def synthesize_text(text, output_file, options=None):
    """Synthesize text to speech and save to a file."""
    if options is None:
        options = {
            "language": "en",
            "voice": "p225",  # Default voice
            "speed": 1.0,
            "format": "wav"
        }
    
    payload = {
        "text": text,
        "options": options
    }
    
    response = requests.post(
        "http://localhost:8000/synthesize",
        json=payload
    )
    
    if response.status_code == 200:
        with open(output_file, "wb") as f:
            f.write(response.content)
        print(f"Successfully synthesized speech to {output_file}")
        print(f"Processing time: {response.headers.get('X-Processing-Time', 'unknown')} seconds")
        return True
    else:
        print(f"Error synthesizing speech: {response.status_code}")
        print(response.text)
        return False

def process_sample_file(input_file, output_dir, options=None):
    """Process a sample text file and generate speech."""
    # Read the input file
    with open(input_file, "r") as f:
        text = f.read().strip()
    
    # Create the output filename
    input_filename = os.path.basename(input_file)
    output_filename = os.path.splitext(input_filename)[0]
    format_ext = options.get("format", "wav") if options else "wav"
    output_file = os.path.join(output_dir, f"{output_filename}.{format_ext}")
    
    # Synthesize the text
    return synthesize_text(text, output_file, options)

def main():
    parser = argparse.ArgumentParser(description="Test the TTS service with sample files.")
    parser.add_argument("--input-dir", default="input", help="Directory containing input text files")
    parser.add_argument("--output-dir", default="output", help="Directory to save output audio files")
    parser.add_argument("--voice", help="Voice to use for synthesis")
    parser.add_argument("--language", default="en", help="Language to use for synthesis")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed multiplier")
    parser.add_argument("--format", default="wav", choices=["wav", "mp3", "ogg"], help="Output audio format")
    parser.add_argument("--list-voices", action="store_true", help="List available voices and exit")
    parser.add_argument("--list-languages", action="store_true", help="List available languages and exit")
    
    args = parser.parse_args()
    
    # Check if the server is running
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code != 200:
            print("Error: TTS service is not running or not healthy.")
            sys.exit(1)
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the TTS service. Make sure it's running on http://localhost:8000")
        sys.exit(1)
    
    # List voices if requested
    if args.list_voices:
        voices = get_available_voices()
        print("Available voices:")
        for voice in voices:
            print(f"  - {voice}")
        sys.exit(0)
    
    # List languages if requested
    if args.list_languages:
        languages = get_available_languages()
        print("Available languages:")
        for language in languages:
            print(f"  - {language}")
        sys.exit(0)
    
    # Get the base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set up input and output directories
    input_dir = os.path.join(base_dir, args.input_dir)
    output_dir = os.path.join(base_dir, args.output_dir)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up synthesis options
    options = {
        "language": args.language,
        "voice": args.voice if args.voice else "p225",  # Default voice
        "speed": args.speed,
        "format": args.format
    }
    
    # Process all text files in the input directory
    input_files = [f for f in os.listdir(input_dir) if f.endswith(".txt")]
    if not input_files:
        print(f"No text files found in {input_dir}")
        sys.exit(1)
    
    print(f"Found {len(input_files)} text files to process")
    
    success_count = 0
    for input_file in input_files:
        input_path = os.path.join(input_dir, input_file)
        print(f"Processing {input_path}...")
        if process_sample_file(input_path, output_dir, options):
            success_count += 1
    
    print(f"Successfully processed {success_count} out of {len(input_files)} files")
    print(f"Output files are in {output_dir}")

if __name__ == "__main__":
    main() 