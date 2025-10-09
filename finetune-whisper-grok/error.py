ai_dev@ppe-nvidia-k8s-worker01:/u01/user-data/vint1/finetune_whisper$ docker exec -it lora-finetuning-job bash
ai_dev@a98c021bfbc1:/app$ ls
Dockerfile         create_tokenizer.py  merge_lora.py         merged_model_dir  my-whisper-medium-lora  requirements.txt  run_workflow.sh  testfile       transcribe_faster_whisper.py
convert_to_ct2.py  finetune_lora.py     merged_model_ct2_dir  model             remove_folder.bash      run_temp.sh       test_audio       transcribe.py
ai_dev@a98c021bfbc1:/app$ python3 finetune_lora.py
Traceback (most recent call last):
  File "/app/finetune_lora.py", line 64, in <module>
    model = PeftModel.from_pretrained(model, ADAPTER_TO_CONTINUE_FROM, is_trainable=True)
  File "/usr/local/lib/python3.10/dist-packages/peft/peft_model.py", line 525, in from_pretrained
    model = cls(
  File "/usr/local/lib/python3.10/dist-packages/peft/peft_model.py", line 132, in __init__
    self.base_model = cls(model, {adapter_name: peft_config}, adapter_name)
  File "/usr/local/lib/python3.10/dist-packages/peft/tuners/lora/model.py", line 142, in __init__
    super().__init__(model, config, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)
  File "/usr/local/lib/python3.10/dist-packages/peft/tuners/tuners_utils.py", line 180, in __init__
    self.inject_adapter(self.model, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)
  File "/usr/local/lib/python3.10/dist-packages/peft/tuners/tuners_utils.py", line 508, in inject_adapter
    self._create_and_replace(peft_config, adapter_name, target, target_name, parent, current_key=key)
  File "/usr/local/lib/python3.10/dist-packages/peft/tuners/lora/model.py", line 237, in _create_and_replace
    new_module = self._create_new_module(lora_config, adapter_name, target, device_map=device_map, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/peft/tuners/lora/model.py", line 318, in _create_new_module
    from .bnb import dispatch_bnb_8bit
  File "/usr/local/lib/python3.10/dist-packages/peft/tuners/lora/bnb.py", line 19, in <module>
    import bitsandbytes as bnb
  File "/usr/local/lib/python3.10/dist-packages/bitsandbytes/__init__.py", line 15, in <module>
    from .nn import modules
  File "/usr/local/lib/python3.10/dist-packages/bitsandbytes/nn/__init__.py", line 21, in <module>
    from .triton_based_modules import (
  File "/usr/local/lib/python3.10/dist-packages/bitsandbytes/nn/triton_based_modules.py", line 6, in <module>
    from bitsandbytes.triton.dequantize_rowwise import dequantize_rowwise
  File "/usr/local/lib/python3.10/dist-packages/bitsandbytes/triton/dequantize_rowwise.py", line 36, in <module>
    def _dequantize_rowwise(
  File "/usr/local/lib/python3.10/dist-packages/triton/runtime/autotuner.py", line 378, in decorator
    return Autotuner(fn, fn.arg_names, configs, key, reset_to_zero, restore_value, pre_hook=pre_hook,
  File "/usr/local/lib/python3.10/dist-packages/triton/runtime/autotuner.py", line 130, in __init__
    self.do_bench = driver.active.get_benchmarker()
  File "/usr/local/lib/python3.10/dist-packages/triton/runtime/driver.py", line 23, in __getattr__
    self._initialize_obj()
  File "/usr/local/lib/python3.10/dist-packages/triton/runtime/driver.py", line 20, in _initialize_obj
    self._obj = self._init_fn()
  File "/usr/local/lib/python3.10/dist-packages/triton/runtime/driver.py", line 9, in _create_driver
    return actives[0]()
  File "/usr/local/lib/python3.10/dist-packages/triton/backends/nvidia/driver.py", line 535, in __init__
    self.utils = CudaUtils()  # TODO: make static
  File "/usr/local/lib/python3.10/dist-packages/triton/backends/nvidia/driver.py", line 89, in __init__
    mod = compile_module_from_src(Path(os.path.join(dirname, "driver.c")).read_text(), "cuda_utils")
  File "/usr/local/lib/python3.10/dist-packages/triton/backends/nvidia/driver.py", line 58, in compile_module_from_src
    cache = get_cache_manager(key)
  File "/usr/local/lib/python3.10/dist-packages/triton/runtime/cache.py", line 277, in get_cache_manager
    return __cache_cls(_base32(key))
  File "/usr/local/lib/python3.10/dist-packages/triton/runtime/cache.py", line 69, in __init__
    os.makedirs(self.cache_dir, exist_ok=True)
  File "/usr/lib/python3.10/os.py", line 215, in makedirs
    makedirs(head, exist_ok=exist_ok)
  File "/usr/lib/python3.10/os.py", line 215, in makedirs
    makedirs(head, exist_ok=exist_ok)
  File "/usr/lib/python3.10/os.py", line 215, in makedirs
    makedirs(head, exist_ok=exist_ok)
  File "/usr/lib/python3.10/os.py", line 225, in makedirs
    mkdir(name, mode)
PermissionError: [Errno 13] Permission denied: '/home/ai_dev'
