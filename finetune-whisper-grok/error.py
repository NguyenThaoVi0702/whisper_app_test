
ai_dev@ppe-nvidia-k8s-worker01:/u01/user-data/vint1/finetune_whisper$ docker exec -it lora-finetuning-job bash
ai_dev@d3877e909cca:/app$ python3 finetune_lora.py
trainable params: 6,553,600 || all params: 815,431,680 || trainable%: 0.8037
Resolving data files: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 105/105 [00:00<00:00, 258907.65it/s]
Loading dataset shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 91/91 [00:00<00:00, 383.02it/s]
Original train size: 634158
Filtered train size: 633414
Original val size: 7500
Filtered val size: 7494
/app/finetune_lora.py:188: FutureWarning: `tokenizer` is deprecated and will be removed in version 5.0.0 for `Seq2SeqTrainer.__init__`. Use `processing_class` instead.
  trainer = Seq2SeqTrainer(
Starting continued training with evaluation...
{'loss': 0.1677, 'grad_norm': 1.0517656803131104, 'learning_rate': 4.800000000000001e-06, 'epoch': 0.0}
{'loss': 0.1376, 'grad_norm': 1.007310152053833, 'learning_rate': 9.800000000000001e-06, 'epoch': 0.01}
{'loss': 0.1068, 'grad_norm': 0.8988784551620483, 'learning_rate': 9.466666666666667e-06, 'epoch': 0.01}
{'loss': 0.0937, 'grad_norm': 0.7448081970214844, 'learning_rate': 8.91111111111111e-06, 'epoch': 0.01}
{'loss': 0.0823, 'grad_norm': 0.6654237508773804, 'learning_rate': 8.355555555555556e-06, 'epoch': 0.01}
{'loss': 0.0869, 'grad_norm': 0.7928948402404785, 'learning_rate': 7.800000000000002e-06, 'epoch': 0.02}
{'loss': 0.0835, 'grad_norm': 0.6111237406730652, 'learning_rate': 7.244444444444445e-06, 'epoch': 0.02}
{'loss': 0.0723, 'grad_norm': 0.6261500716209412, 'learning_rate': 6.688888888888889e-06, 'epoch': 0.02}
{'loss': 0.0708, 'grad_norm': 0.49129214882850647, 'learning_rate': 6.133333333333334e-06, 'epoch': 0.02}
{'loss': 0.0711, 'grad_norm': 0.6761265993118286, 'learning_rate': 5.577777777777778e-06, 'epoch': 0.03}
{'loss': 0.0703, 'grad_norm': 0.593568742275238, 'learning_rate': 5.022222222222223e-06, 'epoch': 0.03}
 58%|███████████████████████████████████████████████████████████████████████████  
