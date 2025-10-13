ai_dev@2158b1d839aa:/app$ python3 finetune_lora.py
trainable params: 6,553,600 || all params: 815,431,680 || trainable%: 0.8037
Resolving data files: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 105/105 [00:00<00:00, 291271.11it/s]
Loading dataset shards: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 91/91 [00:00<00:00, 216.55it/s]
Original train size: 634158
Filtered train size: 633414
Original val size: 7500
Filtered val size: 7494
Traceback (most recent call last):
  File "/app/finetune_lora.py", line 124, in <module>
    metric = evaluate.load("wer")
  File "/usr/local/lib/python3.10/dist-packages/evaluate/loading.py", line 748, in load
    evaluation_module = evaluation_module_factory(
  File "/usr/local/lib/python3.10/dist-packages/evaluate/loading.py", line 681, in evaluation_module_factory
    raise FileNotFoundError(
FileNotFoundError: Couldn't find a module script at /app/wer/wer.py. Module 'wer' doesn't exist on the Hugging Face Hub either.
