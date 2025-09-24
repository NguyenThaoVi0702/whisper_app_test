import os
import glob
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union

from datasets import load_dataset, Audio
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    BitsAndBytesConfig
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

# --- 1. Define Paths and Configuration ---
# This is your STARTING model directory
BASE_MODEL_PATH = "merged_model_ct2_dir"
DATASET_PATH = "viet_bud500"
# This is where the NEW adapter weights will be saved
LORA_ADAPTER_OUTPUT_DIR = "./new-lora-adapters"

LANGUAGE = "Vietnamese"
TASK = "transcribe"
SAMPLING_RATE = 16000

# --- 2. Load Processor from the Base Model ---
# The processor contains the tokenizer and feature extractor
processor = WhisperProcessor.from_pretrained(BASE_MODEL_PATH, language=LANGUAGE, task=TASK)
tokenizer = processor.tokenizer
feature_extractor = processor.feature_extractor

# --- 3. Load and Prepare the Dataset ---
parquet_dir = os.path.join(DATASET_PATH, "data")
data_files = {
    "train": glob.glob(f"{parquet_dir}/train-*.parquet"),
    "validation": glob.glob(f"{parquet_dir}/validation-*.parquet"),
}
dataset = load_dataset('parquet', data_files=data_files).cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
print("Dataset loaded:", dataset)

def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = tokenizer(batch["transcription"]).input_ids
    return batch

dataset = dataset.map(prepare_dataset, remove_columns=dataset["train"].column_names, num_proc=1)

# --- 4. Load Model with 4-bit Quantization (QLoRA) ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
model = WhisperForConditionalGeneration.from_pretrained(
    BASE_MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto"
)
model = prepare_model_for_kbit_training(model)

# --- 5. Configure LoRA Adapters ---
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# --- 6. Define Data Collator ---
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

# --- 7. Define Training Arguments ---
training_args = Seq2SeqTrainingArguments(
    output_dir=LORA_ADAPTER_OUTPUT_DIR,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=5e-6,
    warmup_steps=500,
    max_steps=4000,
    gradient_checkpointing=True,
    fp16=True, # For A100, bf16=True is also a great option
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    remove_unused_columns=False, # Important for PEFT
    label_names=["labels"], # Important for PEFT
    optim="adamw_bnb_8bit",
)

# --- 8. Create Trainer and Start Training ---
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=data_collator,
    tokenizer=processor.feature_extractor,
)
trainer.train()

# --- 9. Save the Final Adapter Weights ---
print(f"Saving final adapter weights to {LORA_ADAPTER_OUTPUT_DIR}")
trainer.save_model()
