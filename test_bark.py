import os
os.environ['TORCH_LOAD_WEIGHTS_ONLY'] = '0'  # Disable weights_only for torch.load

from bark import SAMPLE_RATE, generate_audio, preload_models
import scipy.io.wavfile as wavfile

# Download and load all models
preload_models()

# Generate audio from text
text = "Hello! This is a test of the Bark text to speech system."
audio_array = generate_audio(text)

# Save audio to disk
wavfile.write("bark_generation.wav", SAMPLE_RATE, audio_array)
print("Audio file generated successfully!") 