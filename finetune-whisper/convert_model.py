import os
import shutil
from ctranslate2.converters import TransformersConverter

# --- Configuration ---
SOURCE_MODEL_PATH = "./merged_model_dir"
CT2_MODEL_SAVE_PATH = "./faster-whisper-final-model"
QUANTIZATION = "float16"

print(f"Starting conversion of model from '{SOURCE_MODEL_PATH}'...")
os.makedirs(CT2_MODEL_SAVE_PATH, exist_ok=True)

# --- 1. Run the CTranslate2 Conversion ---
converter = TransformersConverter(SOURCE_MODEL_PATH)
converter.convert(
    output_dir=CT2_MODEL_SAVE_PATH,
    quantization=QUANTIZATION,
)
print("Model weight conversion complete.")

# --- 2. Robustly Copy All Necessary Configuration Files ---
# This new logic copies all essential config/tokenizer files, making the
# final directory complete and ready for offline use.

print(f"Copying all configuration and tokenizer files from '{SOURCE_MODEL_PATH}'...")

# Get a list of all files in the source directory
all_files = os.listdir(SOURCE_MODEL_PATH)

# Define files to exclude (the large PyTorch model weights which are no longer needed)
files_to_exclude = {"pytorch_model.bin", "model.safetensors"}

copied_files_count = 0
for filename in all_files:
    # If the file is NOT in our exclusion list, copy it.
    if filename not in files_to_exclude:
        source_file = os.path.join(SOURCE_MODEL_PATH, filename)
        destination_file = os.path.join(CT2_MODEL_SAVE_PATH, filename)
        
        # Check if it's actually a file and not a directory
        if os.path.isfile(source_file):
            shutil.copy(source_file, destination_file)
            copied_files_count += 1

print(f"Copied {copied_files_count} configuration files.")
print("The final model is now complete and ready for offline inference with faster-whisper.")
