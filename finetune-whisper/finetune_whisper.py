import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import glob  # ### NEW ### Import the glob library to find files

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
MODEL_PATH = "merged_model_dir"
# ### NEW ### Path to the top-level dataset folder
DATASET_PATH = "viet_bud500"
OUTPUT_DIR = "./whisper-finetuned-vi"
LANGUAGE = "Vietnamese"
TASK = "transcribe"

# --- 2. Load Processor, Feature Extractor, and Tokenizer ---
processor = WhisperProcessor.from_pretrained(MODEL_PATH)
feature_extractor = processor.feature_extractor
tokenizer = processor.tokenizer

# --- 3. Load and Prepare the Dataset ---

# ### NEW ### - This whole section is updated to handle the Parquet file structure
# Point to the directory containing the Parquet files
parquet_dir = os.path.join(DATASET_PATH, "data")

# Create lists of files for each split using glob
train_files = glob.glob(f"{parquet_dir}/train-*.parquet")
test_files = glob.glob(f"{parquet_dir}/test-*.parquet")
validation_files = glob.glob(f"{parquet_dir}/validation-*.parquet")

# Create a dictionary telling `load_dataset` where to find the files for each split
data_files = {
    "train": train_files,
    "test": test_files,
    "validation": validation_files,
}

# Load the dataset from the Parquet files
# The first argument 'parquet' tells the library what kind of files to expect.
dataset = load_dataset('parquet', data_files=data_files)

print("Dataset loaded successfully:")
print(dataset)
# ### END OF NEW SECTION ###

# This part remains the same
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = tokenizer(batch["sentence"], language=LANGUAGE, task=TASK).input_ids
    return batch

# You have train, validation, and test splits, so map all of them
dataset["train"] = dataset["train"].map(prepare_dataset, remove_columns=dataset["train"].column_names, num_proc=1)
dataset["test"] = dataset["test"].map(prepare_dataset, remove_columns=dataset["test"].column_names, num_proc=1)
dataset["validation"] = dataset["validation"].map(prepare_dataset, remove_columns=dataset["validation"].column_names, num_proc=1)


# --- 4. Define a Data Collator ---
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
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
    label_ids[label_ids == -100] = tokenizer.pad_token_id
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
    # ### Best Practice ###: Use the 'validation' split for evaluation during training
    eval_dataset=dataset["validation"],
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
