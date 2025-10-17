
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

Starting conversion from './new_whisper_vietbud500_merged_model' to CT2...
Traceback (most recent call last):
  File "/app/convert_to_ct2.py", line 17, in <module>
    converter.convert(
  File "/usr/local/lib/python3.10/dist-packages/ctranslate2/converters/converter.py", line 84, in convert
    raise RuntimeError(
RuntimeError: output directory ./new_whisper_vietbud500_ct2_model already exists, use --force to override
