from transformers import pipeline

# Load the fine-tuned model
pipe = pipeline("automatic-speech-recognition", model="whisper-finetuned-vi")

# Transcribe an audio file
result = pipe("path/to/your/audio.wav")
print(result["text"])
