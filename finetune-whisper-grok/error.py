This container image and its contents are governed by the NVIDIA Deep Learning Container License.
By pulling and using the container, you accept the terms and conditions of this license:
https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.

/app/finetune_lora.py:178: FutureWarning: `tokenizer` is deprecated and will be removed in version 5.0.0 for `Seq2SeqTrainer.__init__`. Use `processing_class` instead.
  trainer = Seq2SeqTrainer(
trainable params: 6,553,600 || all params: 815,431,680 || trainable%: 0.8037
Original train size: 634158
Filtered train size: 633414
Original val size: 7500
Filtered val size: 7494
Starting continued training with evaluation...
  0%|          | 0/5000 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/bitsandbytes/autograd/_functions.py:315: UserWarning: MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
  warnings.warn(f"MatMul8bitLt: inputs will be cast from {A.dtype} to float16 during quantization")
/usr/local/lib/python3.10/dist-packages/bitsandbytes/autograd/_functions.py:315: UserWarning: MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization
  warnings.warn(f"MatMul8bitLt: inputs will be cast from {A.dtype} to float16 during quantization")
 10%|█         | 500/5000 [5:19:05<47:36:18, 38.08s/it]`generation_config` default values have been modified to match model-specific defaults: {'suppress_tokens': [1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61, 62, 63, 90, 91, 92, 93, 359, 503, 522, 542, 873, 893, 902, 918, 922, 931, 1350, 1853, 1982, 2460, 2627, 3246, 3253, 3268, 3536, 3846, 3961, 4183, 4667, 6585, 6647, 7273, 9061, 9383, 10428, 10929, 11938, 12033, 12331, 12562, 13793, 14157, 14635, 15265, 15618, 16553, 16604, 18362, 18956, 20075, 21675, 22520, 26130, 26161, 26435, 28279, 29464, 31650, 32302, 32470, 36865, 42863, 47425, 49870, 50254, 50258, 50359, 50360, 50361, 50362, 50363], 'begin_suppress_tokens': [220, 50257]}. If this is not desired, please set these values explicitly.
The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
A custom logits processor of type <class 'transformers.generation.logits_process.SuppressTokensLogitsProcessor'> has been passed to `.generate()`, but it was also created in `.generate()`, given its parameterization. The custom <class 'transformers.generation.logits_process.SuppressTokensLogitsProcessor'> will take precedence. Please check the docstring of <class 'transformers.generation.logits_process.SuppressTokensLogitsProcessor'> to see related `.generate()` flags.
A custom logits processor of type <class 'transformers.generation.logits_process.SuppressTokensAtBeginLogitsProcessor'> has been passed to `.generate()`, but it was also created in `.generate()`, given its parameterization. The custom <class 'transformers.generation.logits_process.SuppressTokensAtBeginLogitsProcessor'> will take precedence. Please check the docstring of <class 'transformers.generation.logits_process.SuppressTokensAtBeginLogitsProcessor'> to see related `.generate()` flags.
{'loss': 0.1689, 'grad_norm': 1.035498023033142, 'learning_rate': 4.800000000000001e-06, 'epoch': 0.0}
{'loss': 0.1385, 'grad_norm': 0.9220166206359863, 'learning_rate': 9.800000000000001e-06, 'epoch': 0.01}
{'loss': 0.1074, 'grad_norm': 0.8714247941970825, 'learning_rate': 9.951515151515152e-06, 'epoch': 0.01}
{'loss': 0.093, 'grad_norm': 0.7243705987930298, 'learning_rate': 9.901010101010102e-06, 'epoch': 0.01}
{'loss': 0.0808, 'grad_norm': 0.6885049343109131, 'learning_rate': 9.850505050505051e-06, 'epoch': 0.01}

{'loss': 0.0855, 'grad_norm': 0.7983627915382385, 'learning_rate': 9.800000000000001e-06, 'epoch': 0.02}
{'loss': 0.0816, 'grad_norm': 0.6248502731323242, 'learning_rate': 9.749494949494949e-06, 'epoch': 0.02}
{'loss': 0.0701, 'grad_norm': 0.5978335738182068, 'learning_rate': 9.6989898989899e-06, 'epoch': 0.02}
{'loss': 0.0684, 'grad_norm': 0.44836458563804626, 'learning_rate': 9.648484848484849e-06, 'epoch': 0.02}
{'loss': 0.069, 'grad_norm': 0.6478220820426941, 'learning_rate': 9.597979797979798e-06, 'epoch': 0.03}
{'loss': 0.0668, 'grad_norm': 0.5798444151878357, 'learning_rate': 9.547474747474748e-06, 'epoch': 0.03}
{'loss': 0.0676, 'grad_norm': 0.5290008783340454, 'learning_rate': 9.496969696969698e-06, 'epoch': 0.03}
{'loss': 0.0641, 'grad_norm': 0.5988746285438538, 'learning_rate': 9.446464646464648e-06, 'epoch': 0.03}
{'loss': 0.0653, 'grad_norm': 0.45818886160850525, 'learning_rate': 9.395959595959597e-06, 'epoch': 0.04}
{'loss': 0.0654, 'grad_norm': 0.5955553650856018, 'learning_rate': 9.345454545454547e-06, 'epoch': 0.04}
{'loss': 0.0665, 'grad_norm': 0.5939226746559143, 'learning_rate': 9.294949494949495e-06, 'epoch': 0.04}
{'loss': 0.06, 'grad_norm': 0.541419506072998, 'learning_rate': 9.244444444444445e-06, 'epoch': 0.04}
{'loss': 0.0627, 'grad_norm': 0.4390562176704407, 'learning_rate': 9.193939393939395e-06, 'epoch': 0.05}
{'loss': 0.0619, 'grad_norm': 0.5925328135490417, 'learning_rate': 9.143434343434344e-06, 'epoch': 0.05}
 10%|█         | 500/5000 [6:57:02<47:36:18, 38.08s/i/usr/local/lib/python3.10/dist-packages/bitsandbytes/autograd/_functions.py:315: UserWarning: MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
  warnings.warn(f"MatMul8bitLt: inputs will be cast from {A.dtype} to float16 during quantization")
/usr/local/lib/python3.10/dist-packages/bitsandbytes/autograd/_functions.py:315: UserWarning: MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization
  warnings.warn(f"MatMul8bitLt: inputs will be cast from {A.dtype} to float16 during quantization")
 15%|█▌{'eval_loss': 0.058009713888168335, 'eval_wer': 0.03705870309163766, 'eval_runtime': 5877.1367, 'eval_samples_per_second': 1.275, 'eval_steps_per_second': 0.638, 'epoch': 0.05}
{'loss': 0.0575, 'grad_norm': 0.5029078722000122, 'learning_rate': 9.042424242424244e-06, 'epoch': 0.05}
{'loss': 0.0565, 'grad_norm': 0.5483759641647339, 'learning_rate': 8.991919191919193e-06, 'epoch': 0.06}
{'loss': 0.0571, 'grad_norm': 0.5834385752677917, 'learning_rate': 8.941414141414142e-06, 'epoch': 0.06}
{'loss': 0.057, 'grad_norm': 0.5040060877799988, 'learning_rate': 8.890909090909091e-06, 'epoch': 0.06}
{'loss': 0.0639, 'grad_norm': 0.48350515961647034, 'learning_rate': 8.840404040404041e-06, 'epoch': 0.06}
{'loss': 0.0581, 'grad_norm': 0.6504068374633789, 'learning_rate': 8.78989898989899e-06, 'epoch': 0.07}
{'loss': 0.0588, 'grad_norm': 0.5684482455253601, 'learning_rate': 8.73939393939394e-06, 'epoch': 0.07}
{'loss': 0.053, 'grad_norm': 0.4740424156188965, 'learning_rate': 8.68888888888889e-06, 'epoch': 0.07}
{'loss': 0.0562, 'grad_norm': 0.535355806350708, 'learning_rate': 8.63838383838384e-06, 'epoch': 0.07}
{'loss': 0.059, 'grad_norm': 0.3021358251571655, 'learning_rate': 8.587878787878788e-06, 'epoch': 0.08}
{'loss': 0.0546, 'grad_norm': 0.6091563701629639, 'learning_rate': 8.537373737373738e-06, 'epoch': 0.08}
{'loss': 0.0556, 'grad_norm': 0.5204454660415649, 'learning_rate': 8.486868686868687e-06, 'epoch': 0.08}
{'loss': 0.0568, 'grad_norm': 0.6782100796699524, 'learning_rate': 8.436363636363637e-06, 'epoch': 0.08}
{'loss': 0.0556, 'grad_norm': 0.45114970207214355, 'learning_rate': 8.385858585858587e-06, 'epoch': 0.09}
{'loss': 0.0523, 'grad_norm': 0.5228332877159119, 'learning_rate': 8.335353535353537e-06, 'epoch': 0.09}
{'loss': 0.0574, 'grad_norm': 0.46733540296554565, 'learning_rate': 8.284848484848486e-06, 'epoch': 0.09}
{'loss': 0.0493, 'grad_norm': 0.44678160548210144, 'learning_rate': 8.234343434343434e-06, 'epoch': 0.09}
{'loss': 0.0521, 'grad_norm': 0.5333273410797119, 'learning_rate': 8.183838383838384e-06, 'epoch': 0.1}
{'loss': 0.0557, 'grad_norm': 0.4939021170139313, 'learning_rate': 8.133333333333334e-06, 'epoch': 0.1}
{'loss': 0.0575, 'grad_norm': 0.4804300367832184, 'learning_rate': 8.082828282828284e-06, 'epoch': 0.1}
 20%|██        | 1000/5000 [13:54:18<42:29:30, 38.24s/usr/local/lib/python3.10/dist-packages/bitsandbytes/autograd/_functions.py:315: UserWarning: MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
  warnings.warn(f"MatMul8bitLt: inputs will be cast from {A.dtype} to float16 during quantization")
/usr/local/lib/python3.10/dist-packages/bitsandbytes/autograd/_functions.py:315: UserWarning: MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization
  warnings.warn(f"MatMul8bitLt: inputs will be cast from {A.dtype} to float16 during quantization")

