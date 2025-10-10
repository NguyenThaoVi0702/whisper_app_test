ai_dev@67fefbbf1024:/app$ python3 finetune_lora.py
trainable params: 6,553,600 || all params: 815,431,680 || trainable%: 0.8037
Resolving data files: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 105/105 [00:00<00:00, 287656.38it/s]
Loading dataset shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 91/91 [00:00<00:00, 159.61it/s]
Original train size: 634158
Filtered train size: 633414
Original val size: 7500
Filtered val size: 7494
Traceback (most recent call last):
  File "/app/finetune_lora.py", line 162, in <module>
    training_args = Seq2SeqTrainingArguments(
  File "<string>", line 137, in __init__
  File "/usr/local/lib/python3.10/dist-packages/transformers/training_args.py", line 1648, in __post_init__
    raise ValueError(
ValueError: --load_best_model_at_end requires the save and eval strategy to match, but found
- Evaluation strategy: no
- Save strategy: steps
