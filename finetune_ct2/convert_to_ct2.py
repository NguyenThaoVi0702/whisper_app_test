# convert_model.py (Adjusted for Robust CT2 Conversion on Turbo Model)
import os
import shutil
import json
from ctranslate2.converters import TransformersConverter


SOURCE_MODEL_PATH = "./new_whisper_vietbud500_merged_model"
CT2_MODEL_SAVE_PATH = "./new_whisper_vietbud500_ct2_model"
QUANTIZATION = "float16"  

print(f"Starting conversion from '{SOURCE_MODEL_PATH}' to CT2...")
os.makedirs(CT2_MODEL_SAVE_PATH, exist_ok=True)


converter = TransformersConverter(SOURCE_MODEL_PATH)
converter.convert(
    output_dir=CT2_MODEL_SAVE_PATH,
    quantization=QUANTIZATION,
    force=True, 
)
print("Weight conversion complete.")


print(f"Copying config and tokenizer files from '{SOURCE_MODEL_PATH}'...")
all_files = os.listdir(SOURCE_MODEL_PATH)
files_to_exclude = {"pytorch_model.bin", "model.safetensors", "adapter_model.safetensors"}  

copied_files_count = 0
for filename in all_files:
    if filename not in files_to_exclude:
        source_file = os.path.join(SOURCE_MODEL_PATH, filename)
        dest_file = os.path.join(CT2_MODEL_SAVE_PATH, filename)
        if os.path.isfile(source_file):
            shutil.copy(source_file, dest_file)
            copied_files_count += 1

print(f"Copied {copied_files_count} files.")


ct2_config_path = os.path.join(CT2_MODEL_SAVE_PATH, "config.json")
if os.path.exists(ct2_config_path):
    try:
        with open(ct2_config_path, "r") as f:
            ct2_config = json.load(f)
        
        
        if "alignment_heads" not in ct2_config:
            ct2_config["alignment_heads"] = [
                [2, 4],
                [2, 11],
                [3, 3],
                [3, 6],
                [3, 11],
                [3, 14]
            ]
            print("ENHANCED: Added alignment_heads for Turbo.")

        
        if "suppress_ids" not in ct2_config:
            ct2_config["suppress_ids"] = [
                1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61, 62, 63, 90, 91, 92, 93, 359, 503, 522, 542, 873, 893, 902, 918, 922, 931, 1350, 1853, 1982, 2460, 2627, 3246, 3253, 3268, 3536, 3846, 3961, 4183, 4667, 6585, 6647, 7273, 9061, 9383, 10428, 10929, 11938, 12033, 12331, 12562, 13793, 14157, 14635, 15265, 15618, 16553, 16604, 18362, 18956, 20075, 21675, 22520, 26130, 26161, 26435, 28279, 29464, 31650, 32302, 32470, 36865, 42863, 47425, 49870, 50254, 50258, 50359, 50360, 50361, 50362, 50363
            ]
            print("ENHANCED: Added full suppress_ids for Turbo.")

        if "suppress_ids_begin" not in ct2_config:
            ct2_config["suppress_ids_begin"] = [220, 50257]
            print("ENHANCED: Added suppress_ids_begin.")

        if "lang_ids" not in ct2_config:
            ct2_config["lang_ids"] = list(range(50259, 50359)) 
            print("ENHANCED: Added lang_ids for multilingual support.")

        with open(ct2_config_path, "w") as f:
            json.dump(ct2_config, f, indent=2, ensure_ascii=False)
        
        print("Verified/fixed CT2 config.json with turbo settings")
    except json.JSONDecodeError as e:
        print(f"ENHANCED: JSON error in config: {e}—skipping edits.")
else:
    print("Warning: config.json not found in CT2 output.")

print("Conversion complete. Ready for faster-whisper inference.")
