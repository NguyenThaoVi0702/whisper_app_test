import os
from ctranslate2.converters import TransformersConverter

# --- 1. Define Paths ---
# Path to your fine-tuned model from the previous step
FINETUNED_MODEL_PATH = "whisper-finetuned-vi"

# Path where the converted model will be saved
CONVERTED_MODEL_PATH = "faster-whisper-converted-model"

# Create the output directory if it doesn't exist
os.makedirs(CONVERTED_MODEL_PATH, exist_ok=True)

# --- 2. Run the Conversion ---
# You can choose the quantization type for optimization.
# Common options: "float16", "int8"
# "int8" is faster and uses less memory, with a small trade-off in accuracy.
QUANTIZATION = "int8"

converter = TransformersConverter(
    model_name_or_path=FINETUNED_MODEL_PATH,
)

converter.convert(
    output_dir=CONVERTED_MODEL_PATH,
    quantization=QUANTIZATION,
)

print(f"Conversion complete!")
print(f"Your fine-tuned model has been converted to the CTranslate2 format")
print(f"and saved in: {CONVERTED_MODEL_PATH}")

# You also need the tokenizer and processor files for faster-whisper to work correctly.
# Copy them from your original fine-tuned model directory.
import shutil

for filename in ['preprocessor_config.json', 'tokenizer.json', 'vocab.json']:
    source_path = os.path.join(FINETUNED_MODEL_PATH, filename)
    destination_path = os.path.join(CONVERTED_MODEL_PATH, filename)
    if os.path.exists(source_path):
        shutil.copyfile(source_path, destination_path)

print("Tokenizer and processor configuration files copied.")
