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
{'loss': 0.1688, 'grad_norm': 1.0315258502960205, 'learning_rate': 4.800000000000001e-06, 'epoch': 0.0}
{'loss': 0.1382, 'grad_norm': 0.9598658680915833, 'learning_rate': 9.800000000000001e-06, 'epoch': 0.01}
{'loss': 0.1073, 'grad_norm': 0.854949414730072, 'learning_rate': 9.951515151515152e-06, 'epoch': 0.01}
{'loss': 0.0932, 'grad_norm': 0.7394211888313293, 'learning_rate': 9.901010101010102e-06, 'epoch': 0.01}
{'loss': 0.0805, 'grad_norm': 0.6827227473258972, 'learning_rate': 9.850505050505051e-06, 'epoch': 0.01}
