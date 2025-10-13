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
from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training, LoraConfig
import jiwer  # NEW: Use jiwer for offline WER (no HF download needed)

BASE_MODEL_PATH = "./model"  
ADAPTER_TO_CONTINUE_FROM = "./my-whisper-medium-lora"  
DATASET_PATH = "/tmp/viet_bud500"  
NEW_ADAPTER_SAVE_PATH = "./new_whisper_vietbud500_adapter"

processor = WhisperProcessor.from_pretrained(BASE_MODEL_PATH, language="vi", task="transcribe")

# Load model; add optional quantization for memory efficiency on large dataset (50GB).
# Since original training was without quant, it's safe to omit load_in_8bit=True if VRAM allows (~8GB+ for medium).
# But enable it to prevent OOM during eval/generation, especially with increased steps.
use_quantization = True  # Set to False if issues arise
quantization_config = None
if use_quantization:
    from transformers import BitsAndBytesConfig
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.bfloat16,
    )

model = WhisperForConditionalGeneration.from_pretrained(
    BASE_MODEL_PATH,
    device_map="auto",
    use_cache=False,
    torch_dtype=torch.bfloat16,
    quantization_config=quantization_config,
)

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False})

# Optional: If needed, re-apply LoRA config to match original (r=32, targets q_proj/v_proj)
# But since continuing from existing adapter, skip unless modifying.
# lora_config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")
# model = get_peft_model(model, lora_config)

model = PeftModel.from_pretrained(model, ADAPTER_TO_CONTINUE_FROM, is_trainable=True)

# Remove custom hook if using quantization; otherwise, keep for grad flow
if not use_quantization:
    model.model.model.encoder.conv1.register_forward_hook(lambda module, input, output: output.requires_grad_(True))

model.print_trainable_parameters()

parquet_dir = os.path.join(DATASET_PATH, "data")
train_files = {"train": glob.glob(f"{parquet_dir}/train-*.parquet")}
val_files = {"train": glob.glob(f"{parquet_dir}/validation-*.parquet")}  

train_dataset = load_dataset('parquet', data_files=train_files).cast_column("audio", Audio(sampling_rate=16000))["train"]
val_dataset = load_dataset('parquet', data_files=val_files).cast_column("audio", Audio(sampling_rate=16000))["train"]

def filter_data(example):
    audio_len = len(example["audio"]["array"])
    return 16000 < audio_len < 480000 and len(example["transcription"]) > 5

print("Original train size:", len(train_dataset))
train_dataset = train_dataset.filter(filter_data, num_proc=1)
print("Filtered train size:", len(train_dataset))

print("Original val size:", len(val_dataset))
val_dataset = val_dataset.filter(filter_data, num_proc=1)
print("Filtered val size:", len(val_dataset))

def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=16000
    ).input_features[0]
    batch["labels"] = processor.tokenizer(batch["transcription"]).input_ids
    batch["input_length"] = int(len(audio["array"]))
    batch["labels_length"] = int(len(batch["labels"]))
    return batch

train_dataset = train_dataset.map(prepare_dataset, remove_columns=train_dataset.column_names, num_proc=1)
val_dataset = val_dataset.map(prepare_dataset, remove_columns=val_dataset.column_names, num_proc=1)

def filter_inputs(length):
    return 0 < length < 480000 

def filter_labels(length):
    return length < 448  

train_dataset = train_dataset.filter(filter_inputs, input_columns=["input_length"])
train_dataset = train_dataset.filter(filter_labels, input_columns=["labels_length"])
val_dataset = val_dataset.filter(filter_inputs, input_columns=["input_length"])
val_dataset = val_dataset.filter(filter_labels, input_columns=["labels_length"])

train_dataset = train_dataset.remove_columns(["input_length", "labels_length"])
val_dataset = val_dataset.remove_columns(["input_length", "labels_length"])

train_dataset = train_dataset.shuffle(seed=42)

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

# FIXED: Use jiwer directly for WER (offline, no HF load needed)
def compute_metrics(pred):
    pred_ids = pred.predictions
    
    # Handle potential tuple output from generate()
    if isinstance(pred_ids, tuple):
        pred_ids = pred_ids[0]
    
    # Ensure numpy arrays for decoding
    pred_ids = np.array(pred_ids) if not isinstance(pred_ids, np.ndarray) else pred_ids
    label_ids = pred.label_ids
    
    # Replace -100 with pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_ids = np.array(label_ids) if not isinstance(label_ids, np.ndarray) else label_ids
    
    # Decode
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    
    # Compute WER using jiwer (references first, then predictions; handles normalization/empty cases)
    wer = jiwer.wer(label_str, pred_str)
    return {"wer": wer}

# Enhanced training args: Increased max_steps to 5000 for large 50GB dataset (original initial training was 3600 steps on smaller data).
# With effective batch ~64 (16*4), this allows ~2-3 epochs assuming ~100k+ samples. Adjust based on monitoring.
# Since continuing from ~3600 steps of prior adapter, this adds substantial learning on new data.
# Reduced eval batch for stability; added generation kwargs for faster eval.
has_bf16 = torch.cuda.is_bf16_supported()
training_args = Seq2SeqTrainingArguments(
    output_dir=NEW_ADAPTER_SAVE_PATH,
    per_device_train_batch_size=16,  
    per_device_eval_batch_size=2,  # Smaller for large data/eval stability
    gradient_accumulation_steps=4, 
    learning_rate=1e-5,  
    warmup_steps=50,
    max_steps=5000,  # Increased for 50GB data; monitor WER and stop early if plateau
    bf16=has_bf16,
    fp16=not has_bf16,
    optim="adamw_torch",  # Or "adamw_bnb_8bit" if using bitsandbytes heavily
    do_eval=True,
    eval_steps=500,  # More frequent evals with longer training
    save_steps=500,
    save_total_limit=3,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    remove_unused_columns=False,
    label_names=["labels"],
    predict_with_generate=True, 
    dataloader_num_workers=0,  
    generation_max_length=448,
    generation_num_beams=1,  # Greedy for speed; increase to 4 for better WER but slower eval
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    tokenizer=processor, 
    compute_metrics=compute_metrics,  
)

print("Starting continued training with evaluation...")
trainer.train()

# Final explicit eval for WER report
print("Final evaluation:")
final_metrics = trainer.evaluate()
print(f"Final WER: {final_metrics['eval_wer']:.4f}")

print(f"Training complete. Best adapter saved to {NEW_ADAPTER_SAVE_PATH}")
trainer.save_model()
