# Emotion and Style Parameters

## Overview

VoxRaga TTS API now supports emotion and style parameters that allow you to control the emotional tone and speaking style of the synthesized speech. These parameters can be used to make the speech sound more natural and expressive.

## Supported Parameters

### Emotion

The `emotion` parameter controls the emotional tone of the speech. The following emotions are supported:

- `happy`: A cheerful and joyful tone
- `sad`: A melancholic and sorrowful tone
- `angry`: An intense and forceful tone
- `neutral`: A balanced and unemotional tone
- `excited`: An enthusiastic and energetic tone
- `calm`: A peaceful and soothing tone
- `fearful`: A worried and anxious tone
- `surprised`: An astonished and amazed tone

### Style

The `style` parameter controls the speaking style of the speech. The following styles are supported:

- `formal`: A professional and structured manner
- `casual`: A relaxed and informal tone
- `news`: Similar to a news broadcaster
- `storytelling`: A narrative style for telling stories
- `conversational`: A friendly and interactive style
- `instructional`: Clear and directive, suitable for instructions

## Usage

You can specify the emotion and style parameters in the `options` object of your API request:

```json
{
  "text": "The quick brown fox jumps over the lazy dog.",
  "options": {
    "language": "en",
    "voice": "p225",
    "emotion": "happy",
    "style": "conversational"
  }
}
```

## How It Works

The emotion and style parameters work through a combination of approaches:

1. **Speech Parameter Adjustments**: Each emotion automatically adjusts the speech parameters:
   - `happy`: Increases pitch, speed, and volume
   - `sad`: Decreases pitch, speed, and volume
   - `angry`: Increases pitch, speed, and volume with intensity
   - And so on for other emotions

2. **Text Prompts**: Descriptive prompts are added to the text to guide the TTS model
   - Example: "Say this in a happy and cheerful tone: [original text]"

3. **SSML-like Tags**: Special markup tags are added to provide additional guidance
   - These tags help the TTS model understand how to modify its speech characteristics

## Combining Emotions and Styles

You can combine an emotion with a style to create more nuanced speech. For example:

- `happy` + `conversational`: A cheerful, friendly conversation
- `sad` + `formal`: A solemn, professional announcement
- `excited` + `storytelling`: An enthusiastic narrative
- `calm` + `instructional`: Soothing, clear instructions

## Examples

Here are some examples of how to use the emotion and style parameters:

### Python

```python
import requests

url = "http://localhost:8000/synthesize"
payload = {
    "text": "The quick brown fox jumps over the lazy dog.",
    "options": {
        "language": "en",
        "voice": "p225",
        "emotion": "happy",
        "style": "conversational"
    }
}

response = requests.post(url, json=payload)
with open("output.wav", "wb") as f:
    f.write(response.content)
```

### cURL

```bash
curl -X POST "http://localhost:8000/synthesize" \
     -H "Content-Type: application/json" \
     -d '{
       "text": "The quick brown fox jumps over the lazy dog.",
       "options": {
         "language": "en",
         "voice": "p225",
         "emotion": "happy",
         "style": "conversational"
       }
     }' \
     --output output.wav
```

## Limitations

- The effectiveness of emotion and style parameters depends on the underlying TTS model's capabilities
- Some TTS models may not fully support all the speech parameter adjustments
- The SSML-like tags are interpreted differently by different TTS engines
- For best results, use a TTS model specifically designed for expressive speech

## Model Compatibility

Different TTS models have varying levels of support for emotional speech:

- **VITS models**: Generally good support for speed adjustments, limited support for emotional expression
- **Tacotron models**: Better support for expressive speech through text prompts
- **FastSpeech models**: Variable support depending on the specific implementation

## Future Improvements

- Adding more emotions and styles based on user feedback
- Implementing model-specific parameter tuning for better results
- Supporting full SSML for TTS models that can interpret it
- Exploring fine-tuning of models specifically for emotional speech 