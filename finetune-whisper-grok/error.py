Traceback (most recent call last):
  File "/usr/local/lib/python3.10/dist-packages/transformers/utils/import_utils.py", line 1967, in _get_module
    return importlib.import_module("." + module_name, self.__name__)
  File "/usr/lib/python3.10/importlib/__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 1050, in _gcd_import
  File "<frozen importlib._bootstrap>", line 1027, in _find_and_load
  File "<frozen importlib._bootstrap>", line 1006, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 688, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 883, in exec_module
  File "<frozen importlib._bootstrap>", line 241, in _call_with_frames_removed
  File "/usr/local/lib/python3.10/dist-packages/transformers/integrations/bitsandbytes.py", line 21, in <module>
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
  File "/usr/lib/python3.10/os.py", line 225, in makedirs
    mkdir(name, mode)
PermissionError: [Errno 13] Permission denied: '/.triton'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/app/finetune_lora.py", line 59, in <module>
    model = WhisperForConditionalGeneration.from_pretrained(
  File "/usr/local/lib/python3.10/dist-packages/transformers/modeling_utils.py", line 279, in _wrapper
    return func(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/transformers/modeling_utils.py", line 4228, in from_pretrained
    hf_quantizer.validate_environment(
  File "/usr/local/lib/python3.10/dist-packages/transformers/quantizers/quantizer_bnb_4bit.py", line 80, in validate_environment
    from ..integrations import validate_bnb_backend_availability
  File "<frozen importlib._bootstrap>", line 1075, in _handle_fromlist
  File "/usr/local/lib/python3.10/dist-packages/transformers/utils/import_utils.py", line 1955, in __getattr__
    module = self._get_module(self._class_to_module[name])
  File "/usr/local/lib/python3.10/dist-packages/transformers/utils/import_utils.py", line 1969, in _get_module
    raise RuntimeError(
RuntimeError: Failed to import transformers.integrations.bitsandbytes because of the following error (look up to see its traceback):
[Errno 13] Permission denied: '/.triton'
