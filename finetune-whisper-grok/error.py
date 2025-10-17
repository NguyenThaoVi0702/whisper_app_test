{'loss': 0.0429, 'grad_norm': 0.5281497240066528, 'learning_rate': 9.616161616161617e-07, 'epoch': 0.46}
{'loss': 0.0489, 'grad_norm': 0.42886221408843994, 'learning_rate': 9.111111111111113e-07, 'epoch': 0.46}
{'loss': 0.0439, 'grad_norm': 0.545582115650177, 'learning_rate': 8.606060606060607e-07, 'epoch': 0.46}
{'loss': 0.0448, 'grad_norm': 0.47950154542922974, 'learning_rate': 8.101010101010101e-07, 'epoch': 0.46}
{'loss': 0.0444, 'grad_norm': 0.3660105764865875, 'learning_rate': 7.595959595959597e-07, 'epoch': 0.47}
{'loss': 0.0455, 'grad_norm': 0.3397897481918335, 'learning_rate': 7.090909090909092e-07, 'epoch': 0.47}
{'loss': 0.0458, 'grad_norm': 0.48569753766059875, 'learning_rate': 6.585858585858586e-07, 'epoch': 0.47}
{'loss': 0.0457, 'grad_norm': 0.5213415026664734, 'learning_rate': 6.080808080808082e-07, 'epoch': 0.47}
{'loss': 0.0475, 'grad_norm': 0.5959871411323547, 'learning_rate': 5.575757575757576e-07, 'epoch': 0.48}
{'loss': 0.0455, 'grad_norm': 0.48778581619262695, 'learning_rate': 5.070707070707072e-07, 'epoch': 0.48}
{'loss': 0.0453, 'grad_norm': 0.4039919078350067, 'learning_rate': 4.5656565656565663e-07, 'epoch': 0.48}
{'loss': 0.0446, 'grad_norm': 0.4465676248073578, 'learning_rate': 4.0606060606060605e-07, 'epoch': 0.48}
{'loss': 0.0435, 'grad_norm': 0.610541582107544, 'learning_rate': 3.555555555555556e-07, 'epoch': 0.49}
{'loss': 0.0452, 'grad_norm': 0.441824734210968, 'learning_rate': 3.0505050505050505e-07, 'epoch': 0.49}
{'loss': 0.0469, 'grad_norm': 0.4383220672607422, 'learning_rate': 2.545454545454546e-07, 'epoch': 0.49}
{'loss': 0.0449, 'grad_norm': 0.42150408029556274, 'learning_rate': 2.0404040404040406e-07, 'epoch': 0.5}
{'loss': 0.0429, 'grad_norm': 0.4690219461917877, 'learning_rate': 1.5353535353535356e-07, 'epoch': 0.5}
{'loss': 0.0424, 'grad_norm': 0.3969504237174988, 'learning_rate': 1.0303030303030304e-07, 'epoch': 0.5}
{'loss': 0.0453, 'grad_norm': 0.5267592072486877, 'learning_rate': 5.252525252525253e-08, 'epoch': 0.5}
{'loss': 0.0495, 'grad_norm': 0.5208714604377747, 'learning_rate': 2.0202020202020203e-09, 'epoch': 0.51}
100%|██████████| 5000/5000 [67:12:24<00:00, 48.39s/it]
/usr/local/lib/python3.10/dist-packages/bitsandbytes/autograd/_functions.py:315: UserWarning: MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
  warnings.warn(f"MatMul8bitLt: inputs will be cast from {A.dtype} to float16 during quantization")
/usr/local/lib/python3.10/dist-packages/bitsandbytes/autograd/_functions.py:315: UserWarning: MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization
  warnings.warn(f"MatMul8bitLt: inputs will be cast from {A.dtype} to float16 during quantization")
{'eval_loss': 0.04380811005830765, 'eval_wer': 0.02917605077656556, 'eval_runtime': 5292.8991, 'eval_samples_per_second': 1.416, 'eval_steps_per_second': 0.354, 'epoch': 0.51}
{'train_runtime': 241944.7649, 'train_samples_per_second': 1.323, 'train_steps_per_second': 0.021, 'train_loss': 0.05146642076969147, 'epoch': 0.51}
Final evaluation:
100%|██████████| 1874/1874 [1:28:12<00:00,  2.82s/it]t]
Final WER: 0.0292
Training complete. Best adapter saved to ./new_whisper_vietbud500_adapter
