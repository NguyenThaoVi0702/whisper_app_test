from transformers import WhisperForConditionalGeneration, WhisperFeatureExtractor, WhisperTokenizer, BitsAndBytesConfig, Seq2SeqTrainingArguments, Seq2SeqTrainer, WhisperProcessor, AutoTokenizer
from peft import PeftModel, PeftConfig

modelID = "./model"
base_model = WhisperForConditionalGeneration.from_pretrained(
modelID
#, use_cache=False, device_map="auto",  # in case weird bug in `peft`: device_map={"": 0}
#quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
)
FEATURE_EXTRACTOR = WhisperFeatureExtractor.from_pretrained(modelID)
#TOKENIZER = WhisperTokenizer.from_pretrained(modelID, language="vi", task="transcribe")
TOKENIZER = AutoTokenizer.from_pretrained(modelID)

#model = PeftModel.from_pretrained(base_model, "./my-whisper-medium-lora",is_trainable = False)
#model = model.merge_and_unload()
#model.save_pretrained("merged_model_dir")
#FEATURE_EXTRACTOR.save_pretrained("merged_model_dir")
TOKENIZER.save_pretrained("merged_model_ct2_dir")

