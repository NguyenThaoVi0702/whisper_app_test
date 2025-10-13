import os
import glob
from dataclasses import dataclass
import torch
import numpy as np  
from datasets import load_dataset, Audio
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from peft import PeftModel, prepare_model_for_kbit_training  # Removed unused LoraConfig import
import evaluate  # NEW: For robust WER (uses jiwer internally)

# REMOVE: Custom Levenshtein/compute_wer (replaced by evaluate)

BASE_MODEL_PATH = "./model"  # Your STEVE_turbo base
ADAPTER_TO_CONTINUE_FROM = "./my-whisper-medium-lora"  
DATASET_PATH = "/tmp/viet_bud500"  
NEW_ADAPTER_SAVE_PATH = "./new_whisper_vietbud500_adapter"

processor = WhisperProcessor.from_pretrained(BASE_MODEL_PATH, language="vi", task="transcribe")

# Load base: Match previous (full precision) or use 8-bit for VRAM (recommended for large data)
# If sticking to previous style (no quant), comment out load_in_8bit and the prep call below
model = WhisperForConditionalGeneration.from_pretrained(
    BASE_MODEL_PATH,
    device_map="auto",
    use_cache=False,
    torch_dtype=torch.float16,  # Matches config
    load_in_8bit=True,  # Optional: For stability on large data; disable if OOM or mismatch issues
)

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

# Only if using quantization; skip otherwise to match previous code
if 'load_in_8bit' in locals() and load_in_8bit:  # Conditional for flexibility
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False})

model = PeftModel.from_pretrained(model, ADAPTER_TO_CONTINUE_FROM, is_trainable=True)

# NEW: Hook only if no quantization (matches previous code)
if not ('load_in_8bit' in locals() and load_in_8bit):
    model.model.model.encoder.conv1.register_forward_hook(lambda module, input, output: output.requires_grad_(True))

model.print_trainable_parameters()

# ... (dataset loading/filtering/prepare_dataset unchanged; ensure val has 100+ samples for stable WER)

# ... (length filters and shuffling unchanged)

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor  

    def __call__(self, features):
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# NEW: Load evaluate WER metric (like notebook)
metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Handle generate() output (tuple) and shapes (notebook-style robustness)
    if isinstance(pred_ids, tuple):
        pred_ids = pred_ids[0]
    pred_ids = np.where(pred_ids != -100, pred_ids, processor.tokenizer.pad_token_id)  # Mask invalid
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # Decode (no grouping, skip specials)
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    # Compute WER with evaluate (handles jiwer normalization)
    wer = metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# UPDATED: Scaled for large data; epochs over steps for full coverage
training_args = Seq2SeqTrainingArguments(
    output_dir=NEW_ADAPTER_SAVE_PATH,
    per_device_train_batch_size=32,  # Increase for efficiency (adjust down if OOM)
    per_device_eval_batch_size=8,    # Balanced for generation
    gradient_accumulation_steps=2,   # Effective batch ~64
    learning_rate=1e-5,  
    warmup_steps=100,                # Scale up for large data
    num_train_epochs=3,              # NEW: Use epochs for full passes (or max_steps=15000 for ~1 epoch est.)
    # max_steps=15000,              # Alt: If preferring steps
    bf16=torch.cuda.is_bf16_supported(),  # Match previous dynamic
    optim="adamw_torch",             # Standard; use "adamw_bnb_8bit" if quant
    do_eval=True,
    eval_steps=500,                  # Frequent eval for monitoring
    save_steps=1000,                 # Less frequent saves
    save_total_limit=5,              # Keep more for large run
    logging_steps=100,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    remove_unused_columns=False,
    label_names=["labels"],
    predict_with_generate=True,      # Essential for WER via generation
    dataloader_num_workers=2,        # Slight increase for large data
    dataloader_drop_last=True,       # NEW: Avoid partial batches
    generation_max_length=448,       # Bound decoding (notebook tip)
    generation_num_beams=1,          # Greedy for speed
    # resume_from_checkpoint=f"{ADAPTER_TO_CONTINUE_FROM}/checkpoint-3500",  # Uncomment to resume from specific old ckpt
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    tokenizer=processor, 
    compute_metrics=compute_metrics,  # Now enabled with evaluate
)

print("Starting continued training with evaluation...")
trainer.train()

# Final eval for WER report
print("Final evaluation:")
final_metrics = trainer.evaluate()
print(f"Final WER: {final_metrics['eval_wer']:.2f}%")

print(f"Training complete. Best adapter saved to {NEW_ADAPTER_SAVE_PATH}")
trainer.save_model()
