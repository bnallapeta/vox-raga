# Manual Tests for VoxRaga

This directory contains scripts for manual testing of the VoxRaga API. These tests are designed to be run manually to verify the functionality of the API and to generate sample outputs for inspection.

## Available Tests

### Emotion and Style Test

The `test_emotion_style.py` script tests the emotion and style parameters of the TTS API. It sends requests with different emotion and style parameters and saves the resulting audio files for comparison.

#### Usage

1. Make sure the VoxRaga API is running on `http://localhost:8000`.
2. Run the script:

```bash
cd /path/to/vox-raga
python manual_tests/test_emotion_style.py
```

3. Check the output files in the `output/emotion_style_test` directory.

#### Test Cases

The script tests:
- Different emotions (happy, sad, angry, neutral, excited, calm)
- Different styles (formal, casual, news, storytelling, conversational, instructional)
- Combinations of emotions and styles

## Adding New Tests

When adding new manual tests, please follow these guidelines:

1. Create a new Python script in the `manual_tests` directory.
2. Add a descriptive docstring at the top of the file.
3. Make the script executable (`chmod +x script_name.py`).
4. Update this README with information about the new test.
5. Include clear output messages in the script to indicate progress and results.
6. Save test outputs to the `output` directory with appropriate subdirectories.

## Best Practices

- Always include error handling in your test scripts.
- Add appropriate delays between API calls to avoid overwhelming the server.
- Use descriptive filenames for output files.
- Include a baseline or control test case for comparison.
- Print clear progress messages during test execution.

## Directory Structure

- `test_new_endpoints.py`: Python script for testing the new API endpoints
- `test_documentation.md`: Comprehensive documentation of the testing process and results
- `output/`: Directory containing generated audio files and test outputs
  - `*.wav`: Audio files generated from various endpoints
  - `batch_synthesis.zip`: ZIP archive containing multiple audio files
  - `batch_output/`: Directory containing extracted audio files
- `docs/`: Directory containing legacy documentation
  - `test_results.md`: Detailed results of the API testing (superseded by test_documentation.md)
  - `summary.md`: Summary of the testing process (superseded by test_documentation.md)

## Running the Tests

To run the manual tests:

1. Make sure the VoxRaga TTS service is running on `http://localhost:8000`
2. Install the required packages:
   ```bash
   pip install requests websocket-client
   ```
3. Run the test script:
   ```bash
   python test_new_endpoints.py
   ```

## Test Endpoints

The manual tests verify the following endpoints:

1. **Batch Synthesis** (`/batch_synthesize`)
   - Generates multiple audio files and returns them as a ZIP archive

2. **Async Synthesis** (`/synthesize/async`)
   - Submits a job and allows checking status and retrieving results

3. **Simple Synthesis** (`/synthesize`)
   - Generates audio for a single text input

4. **WebSocket Endpoint** (`/synthesize/ws`)
   - Streams audio synthesis over a WebSocket connection

5. **Voices by Language** (`/voices/{language}`)
   - Returns available voices for each language

6. **Default Speaker Behavior**
   - Tests the automatic selection of a speaker when "default" is specified

## Key Findings

The manual tests identified and fixed several issues:

1. **Multi-speaker Model Requirement**: The TTS model requires a specific speaker to be defined
2. **Default Speaker Handling**: Implemented automatic selection of a speaker when "default" is specified
3. **Emotion and Style Parameters**: Identified issues with emotion and style tags not working correctly with the speaker requirement

For detailed results and implementation details, see `test_documentation.md`. 