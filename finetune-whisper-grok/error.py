
ai_dev@ppe-nvidia-k8s-worker01:/u01/user-data/vint1/finetune_whisper$ docker logs -f lora-finetuning-job

==========
== CUDA ==
==========

CUDA Version 12.0.1

Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This container image and its contents are governed by the NVIDIA Deep Learning Container License.
By pulling and using the container, you accept the terms and conditions of this license:
https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.

Traceback (most recent call last):
  File "/app/finetune_lora.py", line 18, in <module>
    logging.set_verbosity_error()
AttributeError: module 'logging' has no attribute 'set_verbosity_error'
