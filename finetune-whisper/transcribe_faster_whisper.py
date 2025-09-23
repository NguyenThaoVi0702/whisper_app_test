from faster_whisper import WhisperModel
import time

# --- 1. Load the Model ---
# Point to the directory containing the CONVERTED CTranslate2 model
MODEL_PATH = "faster-whisper-converted-model"

# You can also specify the device and compute type
# For GPU: device="cuda", compute_type="int8_float16" or "float16"
# For CPU: device="cpu", compute_type="int8"
model = WhisperModel(MODEL_PATH, device="cpu", compute_type="int8")

# --- 2. Transcribe an Audio File ---
audio_file = "path/to/your/audio.wav" # <-- CHANGE THIS to your audio file

print(f"Transcribing {audio_file}...")
start_time = time.time()

# The `transcribe` method returns an iterator
segments, info = model.transcribe(audio_file, beam_size=5, language="vi")

# `info` contains metadata about the transcription
print(f"Detected language '{info.language}' with probability {info.language_probability:.2f}")

# Iterate through the segments and print them
full_text = []
for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
    full_text.append(segment.text)

end_time = time.time()
processing_time = end_time - start_time

print("-" * 30)
print("Full Transcription:", "".join(full_text).strip())
print(f"Processing time: {processing_time:.2f} seconds")
