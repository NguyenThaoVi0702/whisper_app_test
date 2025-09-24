import os
import glob
from dataclasses import dataclass
import torch
from datasets import load_dataset, Audio
from transformers import (
    WhisperForConditionalGeneration,
    WhisperFeatureExtractor,
    WhisperTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training

# --- 1. Define Paths and Configuration ---
BASE_MODEL_PATH = "./model"
# This is the adapter you already trained and want to continue training
ADAPTER_TO_CONTINUE_FROM = "./my-whisper-medium-lora"
DATASET_PATH = "./viet_bud500"
# This is where the newly updated adapter will be saved
NEW_ADAPTER_SAVE_PATH = "./my-whisper-medium-lora-continued"

# --- 2. Load Model, Tokenizer, and Feature Extractor ---
feature_extractor = WhisperFeatureExtractor.from_pretrained(BASE_MODEL_PATH)
tokenizer = WhisperTokenizer.from_pretrained(BASE_MODEL_PATH, language="vi", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(
    BASE_MODEL_PATH, use_cache=False, device_map="auto"
)

# --- 3. Load and Prepare PEFT Model for Continued Training ---
# Prepare the model for k-bit training (even if not using quantization, it's good practice for PEFT)
model = prepare_model_for_kbit_training(model)

print(f"Loading adapter from {ADAPTER_TO_CONTINUE_FROM} to continue training...")
# Load the existing adapter and set it to trainable
model = PeftModel.from_pretrained(model, ADAPTER_TO_CONTINUE_FROM, is_trainable=True)

# Recommended: Enable gradient computation for the conv layers
model.model.model.encoder.conv1.register_forward_hook(lambda module, input, output: output.requires_grad_(True))

print("Trainable parameters:")
model.print_trainable_parameters()

# --- 4. Load and Prepare the Dataset ---
parquet_dir = os.path.join(DATASET_PATH, "data")
data_files = {"train": glob.glob(f"{parquet_dir}/train-*.parquet"), "validation": glob.glob(f"{parquet_dir}/validation-*.parquet")}
dataset = load_dataset('parquet', data_files=data_files).cast_column("audio", Audio(sampling_rate=16000))

def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=16000).input_features[0]
    batch["labels"] = tokenizer(batch["transcription"]).input_ids
    return batch

dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names["train"], num_proc=2)

# --- 5. Define Data Collator ---
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    def __call__(self, features):
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        batch = feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding()

# --- 6. Define Training Arguments ---
training_args = Seq2SeqTrainingArguments(
    output_dir=NEW_ADAPTER_SAVE_PATH,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    learning_rate=1e-5, # Use a smaller learning rate for continued training
    warmup_steps=50,
    max_steps=1000, # Set the number of additional steps
    fp16=True,
    evaluation_strategy="steps",
    eval_steps=200,
    save_steps=200,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss", # PEFT trainer doesn't easily support WER
    greater_is_better=False,
    remove_unused_columns=False,
    label_names=["labels"],
)

# --- 7. Create and Run the Trainer ---
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=data_collator,
    tokenizer=feature_extractor,
)

print("Starting continued training...")
trainer.train()

print(f"Training complete. New adapter saved to {NEW_ADAPTER_SAVE_PATH}")
trainer.save_model()
