import os
from ctranslate2.converters import TransformersConverter

FINETUNED_MODEL_PATH = "final_merged_model"
CONVERTED_MODEL_PATH = "faster-whisper-converted-model"
QUANTIZATION = "float16" # Best for A100

print(f"Converting model from {FINETUNED_MODEL_PATH}...")
converter = TransformersConverter(FINETUNED_MODEL_PATH)
converter.convert(output_dir=CONVERTED_MODEL_PATH, quantization=QUANTIZATION)

print(f"Conversion complete. Model saved to {CONVERTED_MODEL_PATH}")
