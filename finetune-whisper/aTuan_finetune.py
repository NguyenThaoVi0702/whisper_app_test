

from dataclasses import dataclass
import torch
import datasets as hugDS
from transformers import WhisperForConditionalGeneration, WhisperFeatureExtractor, WhisperTokenizer, BitsAndBytesConfig, Seq2SeqTrainingArguments, Seq2SeqTrainer, WhisperProcessor
import peft

# Commented out IPython magic to ensure Python compatibility.
# %cd '/content/drive/My Drive/fine_tune_whisper'

modelID = "./model"
FEATURE_EXTRACTOR = WhisperFeatureExtractor.from_pretrained(modelID)
TOKENIZER = WhisperTokenizer.from_pretrained(modelID, language="vi", task="transcribe")
MODEL = WhisperForConditionalGeneration.from_pretrained(
modelID, use_cache=False, device_map="auto",  # in case weird bug in `peft`: device_map={"": 0}
#quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
)
MODEL.config.forced_decoder_ids = None
MODEL.config.suppress_tokens = []

# naive model parallelism setup to train on multi-GPU with PEFT, see: https://github.com/huggingface/peft/issues/242#issuecomment-1491447956
if torch.cuda.device_count() > 1:
  import accelerate
  DEV_MAP = MODEL.hf_device_map.copy()
  DEV_MAP["model.decoder.embed_tokens"] = DEV_MAP["model.decoder.embed_positions"] = DEV_MAP["proj_out"] = MODEL._hf_hook.execution_device
  accelerate.dispatch_model(MODEL, device_map=DEV_MAP)
  setattr(MODEL, "model_parallel", True)
  setattr(MODEL, "is_parallelizable", True)
  # see my other notebook to use distributed data parallelism for more effective gpu usage

DUMMY_TOKEN = -100

MODEL_BIS = peft.get_peft_model(
  peft.prepare_model_for_kbit_training(MODEL, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}),
  peft.LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=.05, bias="none")# peft.AdaLoraConfig(?)  # higher number of trainable parameters
  )
MODEL_BIS.model.model.encoder.conv1.register_forward_hook(lambda module, input, output: output.requires_grad_(True))  # re-enable grad computation for conv layer

MODEL_BIS.print_trainable_parameters()  # 16 millions = 1% of 1.6 billions params of whisper large

import pandas as pd
df = pd.read_excel('speech_data_edited.xlsx')

SAMPLING_RATE = 16_000
import os
import pandas as pd
from datasets import Dataset, Audio

def load_my_own_data(file_path: str):
  # ??c d? li?u
  df = pd.read_excel(file_path) if file_path.endswith(".xlsx") else pd.read_csv(file_path)
  df = df.rename(columns={"text": "transcription", "path": "audio"})

  # Ki?m tra file t?n t?i
  df = df[df["audio"].apply(lambda x: os.path.exists(x))].reset_index(drop=True)

  # T?o Dataset t? Pandas
  ds = Dataset.from_pandas(df)

  # G?n d?nh nghia c?t audio
  ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))

  return ds



MY_DATA = hugDS.concatenate_datasets([
# load_my_data(path="google/fleurs",                        name="vi_vn", split="train", mode=1),
# load_my_data(path="mozilla-foundation/common_voice_16_1", name="vi",    split="train", mode=2),
# load_my_data(path="vivos",                                              split="train", mode=2),
# load_my_data(path="doof-ferb/fpt_fosd",                                 split="train", mode=0),
# load_my_data(path="doof-ferb/infore1_25hours",                          split="train", mode=0),
# load_my_data(path="doof-ferb/vlsp2020_vinai_100h",                      split="train", mode=0),
# load_my_data(path="doof-ferb/LSVSC",                                    split="train", mode=1),
load_my_own_data("speech_data_edited.xlsx"),  # ?? D? li?u c?a b?n
])

def prepare_dataset(batch):
  audio = batch["audio"]
  batch["input_length"] = len(audio["array"])  # compute input length
  batch["input_features"] = FEATURE_EXTRACTOR(audio["array"], sampling_rate=SAMPLING_RATE).input_features[0]  # compute log-Mel input features
  batch["labels"] = TOKENIZER(batch["transcription"]).input_ids  # encode target text to label ids
  batch["labels_length"] = len(batch["labels"])  # compute labels length
  return batch

def filter_inputs(input_length):
  """Filter inputs with zero input length or longer than 30s"""
  return 0 < input_length < 48e4  # 30s ? 16kHz

def filter_labels(labels_length):
  """Filter label sequences longer than max length 448 tokens"""
  return labels_length < 448  # MODEL.config.max_length

# MY_DATA = (MY_DATA
# # .shuffle(seed=42)  # useless coz streaming multiple datasets (cannot set buffer too high coz not enough RAM)
# .map(prepare_dataset)  # no `num_proc` coz streaming
# .filter(filter_inputs, input_columns= ["input_length"], remove_columns= ["input_length"])
# .filter(filter_labels, input_columns=["labels_length"], remove_columns=["labels_length"])
# )  # TODO: enable `batched=True` but don?t know how to write functions

print(MY_DATA)

MY_DATA = (
          MY_DATA
          .map(prepare_dataset)
          .filter(filter_inputs, input_columns=["input_length"])
          .filter(filter_labels, input_columns=["labels_length"])
          .remove_columns(["input_length", "labels_length"])
          )


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
  def __call__(self, features):
    # split inputs and labels since they have to be of different lengths and need different padding methods
    input_features = [{"input_features": feature["input_features"]} for feature in features]
    label_features = [{"input_ids"     : feature["labels"]        } for feature in features]  # get the tokenized label sequences
    
    batch = FEATURE_EXTRACTOR.pad(input_features, return_tensors="pt")  # treat the audio inputs by simply returning torch tensors
    labels_batch =  TOKENIZER.pad(label_features, return_tensors="pt")  # pad the labels to max length
    labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), DUMMY_TOKEN)  # replace padding with -100 to ignore loss correctly
    
    if (labels[:, 0] == TOKENIZER.bos_token_id).all().cpu().item():  # if bos token is appended in previous tokenization step,
      labels = labels[:, 1:]  # cut bos token here as it?s append later anyways
    
    batch["labels"] = labels
    return batch

DATA_COLLATOR = DataCollatorSpeechSeq2SeqWithPadding()
DATA_COLLATOR

#torch.backends.cuda.matmul.allow_tf32 = False
#torch.backends.cudnn.allow_tf32 = False
SAVE_PATH = "./my-whisper-medium-lora"  # mount gdrive using GUI before training
BATCH_SIZE = 2  # should be a power of 2

# colab free tier can only run for 8-12h max daily
# kaggle free tier can only run for 30h max weekly but max 12h per session

has_bf16 = torch.cuda.is_bf16_supported()  # GPU Ampere or later

TRAINING_ARGS = Seq2SeqTrainingArguments(
output_dir=SAVE_PATH,
per_device_train_batch_size=BATCH_SIZE,
# per_device_eval_batch_size=BATCH_SIZE,
fp16=not has_bf16,
bf16=has_bf16,
tf32=has_bf16,
# torch_compile=True,  # SDPA not support whisper yet
report_to=["tensorboard"],

max_steps=3600,  # no `num_train_epochs` coz streaming
logging_steps=25,
save_steps=50,
# eval_steps=50,
# evaluation_strategy="no",  # "steps"
  do_eval=True,
save_total_limit=3,

optim="adamw_bnb_8bit",  # 8-bit AdamW optimizer: lower vram usage than default AdamW
learning_rate=5e-6,
warmup_ratio=.05,  # keep between 5-15%
gradient_accumulation_steps=1 if BATCH_SIZE >= 8 else 8 // BATCH_SIZE,
remove_unused_columns=False, label_names=["labels"],  # required by PEFT
# predict_with_generate=True,  # must disable coz PEFT
)

TRAINER = Seq2SeqTrainer(
args=TRAINING_ARGS,
model=MODEL_BIS,
train_dataset=MY_DATA,
data_collator=DATA_COLLATOR,
# compute_metrics=compute_metrics,  # must disable coz PEFT
tokenizer=FEATURE_EXTRACTOR,  # not TOKENIZER
callbacks = [],
)

TRAINER.train()

TRAINER.save_model()
