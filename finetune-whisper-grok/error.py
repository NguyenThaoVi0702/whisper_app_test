
ai_dev@ppe-nvidia-k8s-worker01:~$ docker logs -f lora-finetuning-job

==========
== CUDA ==
==========

CUDA Version 12.0.1

Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This container image and its contents are governed by the NVIDIA Deep Learning Container License.
By pulling and using the container, you accept the terms and conditions of this license:
https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.

trainable params: 6,553,600 || all params: 815,431,680 || trainable%: 0.8037
Downloading data: 100%|██████████| 105/105 [00:00<00:00, 17882.16files/s]
Generating train split: 634158 examples [03:04, 3443.19 examples/s]
Generating train split: 7500 examples [00:02, 3360.83 examples/s]
Original train size: 634158
Filter: 100%|██████████| 634158/634158 [03:18<00:00, 3189.55 examples/s]
Filtered train size: 633414
Original val size: 7500
Filter: 100%|██████████| 7500/7500 [00:02<00:00, 3228.61 examples/s]
Filtered val size: 7494
Map: 100%|██████████| 633414/633414 [1:44:35<00:00, 100.93 examples/s]s]]
Map: 100%|██████████| 7494/7494 [01:12<00:00, 103.95 examples/s]
Filter:   0%|          | 0/633414 [00:00<?, ? examples/s]
Traceback (most recent call last):
  File "/app/finetune_lora.py", line 120, in <module>
    train_dataset = train_dataset.filter(filter_inputs, input_columns=["input_length"])
  File "/usr/local/lib/python3.10/dist-packages/datasets/arrow_dataset.py", line 557, in wrapper
    out: Union["Dataset", "DatasetDict"] = func(self, *args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/datasets/fingerprint.py", line 442, in wrapper
    out = func(dataset, *args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/datasets/arrow_dataset.py", line 3746, in filter
    indices = self.map(
  File "/usr/local/lib/python3.10/dist-packages/datasets/arrow_dataset.py", line 557, in wrapper
    out: Union["Dataset", "DatasetDict"] = func(self, *args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/datasets/arrow_dataset.py", line 3079, in map
    for rank, done, content in Dataset._map_single(**dataset_kwargs):
  File "/usr/local/lib/python3.10/dist-packages/datasets/arrow_dataset.py", line 3525, in _map_single
    for i, batch in iter_outputs(shard_iterable):
  File "/usr/local/lib/python3.10/dist-packages/datasets/arrow_dataset.py", line 3475, in iter_outputs
    yield i, apply_function(example, i, offset=offset)
  File "/usr/local/lib/python3.10/dist-packages/datasets/arrow_dataset.py", line 3398, in apply_function
    processed_inputs = function(*fn_args, *additional_args, **fn_kwargs)
  File "/usr/local/lib/python3.10/dist-packages/datasets/arrow_dataset.py", line 6443, in get_indices_from_mask_function
    mask.append(function(*input, *additional_args, **fn_kwargs))
  File "/app/finetune_lora.py", line 115, in filter_inputs
    return 0 < example["input_length"] < 480000  # ~30s at 16kHz
