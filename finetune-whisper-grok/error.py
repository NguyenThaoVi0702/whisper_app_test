
trainable params: 6,553,600 || all params: 815,431,680 || trainable%: 0.8037
Traceback (most recent call last):
  File "/app/finetune_lora.py", line 85, in <module>
    train_dataset = load_dataset('parquet', data_files=train_files).cast_column("audio", Audio(sampling_rate=16000))["train"]
  File "/usr/local/lib/python3.10/dist-packages/datasets/load.py", line 2062, in load_dataset
    builder_instance = load_dataset_builder(
  File "/usr/local/lib/python3.10/dist-packages/datasets/load.py", line 1819, in load_dataset_builder
    builder_instance: DatasetBuilder = builder_cls(
  File "/usr/local/lib/python3.10/dist-packages/datasets/builder.py", line 386, in __init__
    os.makedirs(self._cache_dir_root, exist_ok=True)
  File "/usr/lib/python3.10/os.py", line 215, in makedirs
    makedirs(head, exist_ok=exist_ok)
  File "/usr/lib/python3.10/os.py", line 215, in makedirs
    makedirs(head, exist_ok=exist_ok)
  File "/usr/lib/python3.10/os.py", line 225, in makedirs
    mkdir(name, mode)
PermissionError: [Errno 13] Permission denied: '/.cache'
