
ai_dev@67fefbbf1024:/app$ python3 finetune_lora.py
trainable params: 6,553,600 || all params: 815,431,680 || trainable%: 0.8037
Resolving data files: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 105/105 [00:00<00:00, 354305.65it/s]
Loading dataset shards: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 91/91 [00:00<00:00, 1569.70it/s]
Original train size: 634158
Filtered train size: 633414
Original val size: 7500
Filtered val size: 7494
/app/finetune_lora.py:188: FutureWarning: `tokenizer` is deprecated and will be removed in version 5.0.0 for `Seq2SeqTrainer.__init__`. Use `processing_class` instead.
  trainer = Seq2SeqTrainer(
Starting continued training with evaluation...
  0%|                                    
