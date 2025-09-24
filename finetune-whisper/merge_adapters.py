from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel

# --- Configuration ---
# The model you started training with
BASE_MODEL_PATH = "merged_model_ct2_dir"
# The new adapters you just trained
LORA_ADAPTER_PATH = "./new-lora-adapters"
# The final, fully merged model for conversion
FINAL_MODEL_SAVE_PATH = "final_merged_model"

print("Loading base model...")
base_model = WhisperForConditionalGeneration.from_pretrained(
    BASE_MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.float16,
)

print("Loading and merging LoRA adapters...")
# Load the LoRA model and merge
model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
model = model.merge_and_unload()
print("Merging complete.")

# Load the original processor to save alongside the merged model
processor = WhisperProcessor.from_pretrained(BASE_MODEL_PATH)

print(f"Saving the final merged model to {FINAL_MODEL_SAVE_PATH}")
model.save_pretrained(FINAL_MODEL_SAVE_PATH)
processor.save_pretrained(FINAL_MODEL_SAVE_PATH)
print("Final model saved and ready for conversion.")
