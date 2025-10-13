
ai_dev@2158b1d839aa:/app$ python3 finetune_lora.py
trainable params: 6,553,600 || all params: 815,431,680 || trainable%: 0.8037
Resolving data files: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 105/105 [00:00<00:00, 317521.21it/s]
Loading dataset shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 91/91 [00:00<00:00, 1684.09it/s]
Original train size: 634158
Filtered train size: 633414
Original val size: 7500
Filtered val size: 7494
/app/finetune_lora.py:187: FutureWarning: `tokenizer` is deprecated and will be removed in version 5.0.0 for `Seq2SeqTrainer.__init__`. Use `processing_class` instead.
  trainer = Seq2SeqTrainer(
Starting continued training with evaluation...
  0%|                                                                                                                                        | 0/5000 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/bitsandbytes/autograd/_functions.py:315: UserWarning: MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
  warnings.warn(f"MatMul8bitLt: inputs will be cast from {A.dtype} to float16 during quantization")
/usr/local/lib/python3.10/dist-packages/bitsandbytes/autograd/_functions.py:315: UserWarning: MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization
  warnings.warn(f"MatMul8bitLt: inputs will be cast from {A.dtype} to float16 during quantization")
