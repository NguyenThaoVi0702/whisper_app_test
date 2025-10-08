from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel
import json
import os

BASE_MODEL_PATH = "./model"
ADAPTER_PATH = "./new_whisper_vietbud500_adpapter"
MERGED_MODEL_SAVE_PATH = "./new_whisper_vietbud500_merged_model"

print(f"Loading base model from: {BASE_MODEL_PATH}")
base_model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL_PATH, device_map="auto")

print(f"Loading and merging LoRA adapter from: {ADAPTER_PATH}")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model = model.merge_and_unload()

# Critical: Re-apply config tweaks post-merge to prevent repetitive outputs
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

processor = WhisperProcessor.from_pretrained(BASE_MODEL_PATH, language="vi", task="transcribe")

print(f"Saving merged model, processor, and config to: {MERGED_MODEL_SAVE_PATH}")
model.save_pretrained(MERGED_MODEL_SAVE_PATH)
processor.save_pretrained(MERGED_MODEL_SAVE_PATH)

# --- Offline Fix for Faster-Whisper Alignment Heads Error (Turbo-Specific) ---
# Faster-Whisper requires 'alignment_heads' in config.json for CTC alignment.
# For Whisper Large V3 Turbo (4 decoder layers), use coworker's turbo-tuned heads.
config_path = os.path.join(MERGED_MODEL_SAVE_PATH, "config.json")
if os.path.exists(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    # Add or update alignment_heads (Turbo-specific: 6 pairs for 4-layer decoder)
    if "alignment_heads" not in config:
        config["alignment_heads"] = [
            [2, 4],
            [2, 11],
            [3, 3],
            [3, 6],
            [3, 11],
            [3, 14]
        ]
    
    # Ensure other CT2-friendly keys (if missing)
    if "ctc_loss_reduction" not in config:
        config["ctc_loss_reduction"] = "mean"
    if "ctc_zero_infinity" not in config:
        config["ctc_zero_infinity"] = True
    if "do_sample" not in config:  # For generation stability
        config["do_sample"] = False
    if "num_beams" not in config:
        config["num_beams"] = 1
    if "use_cache" not in config:
        config["use_cache"] = True
    
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"Fixed config.json with turbo alignment_heads and other keys at {config_path}")
else:
    print("Warning: config.json not found—check merge path.")

print(f"Merging complete. Model ready in {MERGED_MODEL_SAVE_PATH}")
