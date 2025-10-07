
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
Downloading data: 100%|██████████| 105/105 [00:00<00:00, 18320.31files/s]
Generating train split: 634158 examples [03:06, 3405.32 examples/s]
Generating train split: 7500 examples [00:02, 3192.14 examples/s]
Original train size: 634158
Filter (num_proc=4): 100%|██████████| 634158/634158 [00:56<00:00, 11240.08 examples/s]
Filtered train size: 633414
Original val size: 7500
Filter (num_proc=4): 100%|██████████| 7500/7500 [00:01<00:00, 4260.01 examples/s]
Filtered val size: 7494
