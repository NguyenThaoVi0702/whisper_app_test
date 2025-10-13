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


def levenshtein_distance(s1, s2):
    """Compute Levenshtein distance between two strings (word-level)."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def compute_wer(reference, hypothesis):
    """Compute Word Error Rate (WER) between lists of strings."""
    if len(reference) == 0 or len(hypothesis) == 0:
        return 1.0 if len(reference) != len(hypothesis) else 0.0  # Handle empty cases
    total_distance = sum(levenshtein_distance(ref.split(), hyp.split()) for ref, hyp in zip(reference, hypothesis))
    total_words = sum(len(ref.split()) for ref in reference)
    return total_distance / total_words if total_words > 0 else 0.0


BASE_MODEL_PATH = "./model"  
ADAPTER_TO_CONTINUE_FROM = "./my-whisper-medium-lora"  
DATASET_PATH = "/tmp/viet_bud500"  
NEW_ADAPTER_SAVE_PATH = "./new_whisper_vietbud500_adapter"



processor = WhisperProcessor.from_pretrained(BASE_MODEL_PATH, language="vi", task="transcribe")

model = WhisperForConditionalGeneration.from_pretrained(
    BASE_MODEL_PATH,
    device_map="auto",
    use_cache=False,
    torch_dtype=torch.bfloat16  
)


model.config.forced_decoder_ids = None
model.config.suppress_tokens = []


model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False})
model = PeftModel.from_pretrained(model, ADAPTER_TO_CONTINUE_FROM, is_trainable=True)


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


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids


    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    wer = compute_wer(label_str, pred_str)
    return {"wer": wer}


training_args = Seq2SeqTrainingArguments(
    output_dir=NEW_ADAPTER_SAVE_PATH,
    per_device_train_batch_size=16,  
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4, 
    learning_rate=1e-5,  
    warmup_steps=50,
    max_steps=500,  
    bf16=True,  
    optim="adamw_torch",  
    do_eval=True,
    eval_steps=200,
    save_steps=200,
    save_total_limit=3,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=False,
    metric_for_best_model="wer",
    greater_is_better=False,
    remove_unused_columns=False,
    label_names=["labels"],
    predict_with_generate=True, 
    dataloader_num_workers=0,  
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

print(f"Training complete. Best adapter saved to {NEW_ADAPTER_SAVE_PATH}")
trainer.save_model()
