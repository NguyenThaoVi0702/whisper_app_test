import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch
import evaluate
from datasets import load_dataset, Audio
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

# --- 1. Define Paths and Model Details ---
MODEL_PATH = "merged_model_dir"  # Path to your local model directory
DATASET_PATH = "viet_bud500_dataset"  # Path to your local dataset directory
OUTPUT_DIR = "./whisper-finetuned-vi"
LANGUAGE = "Vietnamese"
TASK = "transcribe"

# --- 2. Load Processor, Feature Extractor, and Tokenizer ---
# The processor combines the feature extractor and tokenizer
processor = WhisperProcessor.from_pretrained(MODEL_PATH)
feature_extractor = processor.feature_extractor
tokenizer = processor.tokenizer

# --- 3. Load and Prepare the Dataset ---
# This assumes your dataset is structured for the `datasets` library to load
# It will resample the audio to the required 16kHz
dataset = load_dataset(DATASET_PATH)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

def prepare_dataset(batch):
    # Process audio to log-mel spectrogram
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # Tokenize the transcriptions
    batch["labels"] = tokenizer(batch["sentence"], language=LANGUAGE, task=TASK).input_ids
    return batch

dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names["train"], num_proc=1)

# --- 4. Define a Data Collator ---
# This class handles padding for the input features and labels
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 to ignore in loss computation
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # Handle cases where the model starts with a bos token
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# --- 5. Define Evaluation Metrics ---
metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with the pad token ID
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # Decode predictions and labels
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# --- 6. Load the Pre-trained Model ---
model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH)
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

# --- 7. Define Training Arguments ---
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=4000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)

# --- 8. Create the Trainer ---
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

# --- 9. Start Training ---
trainer.train()

# --- 10. Save the Final Model ---
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

print(f"Fine-tuning complete. Model and processor saved to {OUTPUT_DIR}")
