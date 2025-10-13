ai_dev@2158b1d839aa:/app$ python3 finetune_lora.py
trainable params: 6,553,600 || all params: 815,431,680 || trainable%: 0.8037
Resolving data files: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 105/105 [00:00<00:00, 142019.32it/s]
Loading dataset shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 91/91 [00:00<00:00, 1681.34it/s]
Original train size: 634158
Filtered train size: 633414
Original val size: 7500
Filtered val size: 7494
Traceback (most recent call last):
  File "/app/finetune_lora.py", line 157, in <module>
    training_args = Seq2SeqTrainingArguments(
TypeError: Seq2SeqTrainingArguments.__init__() got an unexpected keyword argument 'evaluation_strategy'
