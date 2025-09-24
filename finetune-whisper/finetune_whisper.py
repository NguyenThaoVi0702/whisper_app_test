import os
import glob
from dataclasses import dataclass

import torch
import datasets as hugDS
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

# --- 1. Configuration ---
# Your already fine-tuned model is the new base model
MODEL_PATH = "./merged_model_ct2_dir"
DATASET_PATH = "./viet_bud500"
# The output will be a new, fully updated model
OUTPUT_DIR = "./fully-finetuned-model-v2"

# --- 2. Load Model and Processor ---
# This is much simpler now. Load everything directly from your merged model directory.
print(f"Loading model and processor from: {MODEL_PATH}")
model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH, device_map="auto")
processor = WhisperProcessor.from_pretrained(MODEL_PATH)
feature_extractor = processor.feature_extractor
tokenizer = processor.tokenizer

# --- 3. Load and Prepare the viet_bud500 Dataset ---
print(f"Loading viet_bud500 dataset from: {DATASET_PATH}")
parquet_dir = os.path.join(DATASET_PATH, "data")
train_files = glob.glob(f"{parquet_dir}/train-*.parquet")
validation_files = glob.glob(f"{parquet_dir}/validation-*.parquet")

data_files = {
    "train": train_files,
    "validation": validation_files,
}
dataset = hugDS.load_dataset('parquet', data_files=data_files)
dataset = dataset.cast_column("audio", hugDS.Audio(sampling_rate=16000))

def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=16000).input_features[0]
    batch["labels"] = tokenizer(batch["transcription"]).input_ids
    return batch

print("Processing dataset...")
processed_dataset = dataset.map(prepare_dataset, remove_columns=dataset["train"].column_names)

# --- 4. Define Data Collator ---
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: any
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

# --- 5. Define Training Arguments ---
# IMPORTANT: Use a much smaller learning rate for full fine-tuning
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2, # Accumulate gradients to simulate a larger batch size
    learning_rate=1e-6, # CRITICAL: Very low learning rate
    warmup_steps=50,
    max_steps=1000,
    fp16=True,
    evaluation_strategy="steps",
    eval_steps=200,
    save_steps=200,
    logging_steps=25,
    save_total_limit=3,
    report_to=["tensorboard"],
)

# --- 6. Create and Run the Trainer ---
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=processed_dataset["train"],
    eval_dataset=processed_dataset["validation"],
    data_collator=data_collator,
    tokenizer=processor.feature_extractor,
)

print("Starting FULL fine-tuning from merged model...")
trainer.train()

print("Training complete.")
trainer.save_model()
print(f"New fully fine-tuned model saved to: {OUTPUT_DIR}")
