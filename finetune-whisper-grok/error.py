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

trainable params: 6,553,600 || all params: 815,431,680 || trainable%: 0.8037
Original train size: 634158
Filtered train size: 633414
Original val size: 7500
Filtered val size: 7494
Traceback (most recent call last):
  File "/app/finetune_lora.py", line 143, in <module>
    training_args = Seq2SeqTrainingArguments(
TypeError: Seq2SeqTrainingArguments.__init__() got an unexpected keyword argument 'generation_kwargs'
