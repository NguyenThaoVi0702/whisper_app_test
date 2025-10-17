from transformers import WhisperProcessor
import os

# Paths: Adjust as needed
BASE_MODEL_PATH = "./model"  # Your original base model dir
TARGET_DIR = "./new_whisper_vietbud500_ct2_model"  # Or merged_model dir; ensure it exists

print(f"Loading processor from base model: {BASE_MODEL_PATH}")
processor = WhisperProcessor.from_pretrained(BASE_MODEL_PATH, language="vi", task="transcribe")

print(f"Saving tokenizer files to: {TARGET_DIR}")
os.makedirs(TARGET_DIR, exist_ok=True)
processor.save_pretrained(TARGET_DIR)

# Verify saved files
saved_files = os.listdir(TARGET_DIR)
tokenizer_files = [f for f in saved_files if f.endswith(('.json', '.txt')) and 'tokenizer' in f.lower()]
print(f"Saved tokenizer-related files: {tokenizer_files}")
print("Tokenizer files (tokenizer.json, vocab.json, merges.txt) now available in the target dir.")
