import os
import shutil
from ctranslate2.converters import TransformersConverter

# --- Configuration ---
# This is the standalone model created by the merging script
SOURCE_MODEL_PATH = "./merged_model_dir"
# This is the final directory for your faster-whisper model
CT2_MODEL_SAVE_PATH = "./faster-whisper-final-model"
# Quantization for A100 GPU. Use "int8" for smaller size and faster CPU inference.
QUANTIZATION = "float16"

print(f"Starting conversion of model from '{SOURCE_MODEL_PATH}'...")

# Create the output directory if it doesn't exist
os.makedirs(CT2_MODEL_SAVE_PATH, exist_ok=True)

# Initialize the converter
converter = TransformersConverter(SOURCE_MODEL_PATH)

# Run the conversion
converter.convert(
    output_dir=CT2_MODEL_SAVE_PATH,
    quantization=QUANTIZATION,
)

print(f"Model conversion complete. Output saved to '{CT2_MODEL_SAVE_PATH}'.")

# --- Copy necessary tokenizer/processor files ---
print("Copying tokenizer and preprocessor configuration files...")
files_to_copy = [
    "preprocessor_config.json",
    "tokenizer.json",
    "vocab.json"
]

for filename in files_to_copy:
    source_file = os.path.join(SOURCE_MODEL_PATH, filename)
    if os.path.exists(source_file):
        shutil.copy(source_file, CT2_MODEL_SAVE_PATH)

print("All necessary files have been prepared for faster-whisper.")
