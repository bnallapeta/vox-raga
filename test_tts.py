from TTS.api import TTS

# Initialize TTS with the desired model
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")

# Text to speech
text = "Hello! This is a test of the text to speech system."
output_path = "tts_output.wav"

# Generate the audio
tts.tts_to_file(text=text, file_path=output_path)
print(f"Audio has been generated and saved to {output_path}") 