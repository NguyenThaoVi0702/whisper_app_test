import os
from transformers import WhisperForConditionalGeneration, WhisperFeatureExtractor, AutoTokenizer
from peft import PeftModel

# --- Configuration ---
BASE_MODEL_PATH = "./model"
ADAPTER_PATH = "./my-whisper-medium-lora-continued"
MERGED_MODEL_SAVE_PATH = "./merged_model_dir"

print(f"Loading base model from: {BASE_MODEL_PATH}")
base_model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL_PATH, device_map="auto")

print(f"Loading LoRA adapter from: {ADAPTER_PATH}")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

print("Merging adapter into the base model...")
model = model.merge_and_unload()
print("Merging complete.")

# --- THE DEFINITIVE FIX IS HERE ---
# We use AutoTokenizer to ensure the complete tokenizer, including the
# essential 'tokenizer.json' file, is loaded and saved correctly.
print("Loading tokenizer using AutoTokenizer for a complete save...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
feature_extractor = WhisperFeatureExtractor.from_pretrained(BASE_MODEL_PATH)

print(f"Saving the complete, merged model to: {MERGED_MODEL_SAVE_PATH}")
model.save_pretrained(MERGED_MODEL_SAVE_PATH)
tokenizer.save_pretrained(MERGED_MODEL_SAVE_PATH)
feature_extractor.save_pretrained(MERGED_MODEL_SAVE_PATH)

print(f"Process finished. Your merged model in '{MERGED_MODEL_SAVE_PATH}' is now complete.")
