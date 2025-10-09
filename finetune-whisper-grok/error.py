
Original train size: 634158
Filter:   0%|                                                                                                                                                             | 0/634158 [00:00<?, ? examples/s]
Traceback (most recent call last):
  File "/app/finetune_lora.py", line 85, in <module>
    train_dataset = train_dataset.filter(filter_data, num_proc=1)
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
  File "/usr/local/lib/python3.10/dist-packages/datasets/arrow_dataset.py", line 6423, in get_indices_from_mask_function
    num_examples = len(batch[next(iter(batch.keys()))])
  File "/usr/local/lib/python3.10/dist-packages/datasets/formatting/formatting.py", line 280, in __getitem__
    value = self.format(key)
  File "/usr/local/lib/python3.10/dist-packages/datasets/formatting/formatting.py", line 380, in format
    return self.formatter.format_column(self.pa_table.select([key]))
  File "/usr/local/lib/python3.10/dist-packages/datasets/formatting/formatting.py", line 460, in format_column
    column = self.python_features_decoder.decode_column(column, pa_table.column_names[0])
  File "/usr/local/lib/python3.10/dist-packages/datasets/formatting/formatting.py", line 226, in decode_column
    return self.features.decode_column(column, column_name) if self.features else column
  File "/usr/local/lib/python3.10/dist-packages/datasets/features/features.py", line 2122, in decode_column
    [decode_nested_example(self[column_name], value) if value is not None else None for value in column]
  File "/usr/local/lib/python3.10/dist-packages/datasets/features/features.py", line 2122, in <listcomp>
    [decode_nested_example(self[column_name], value) if value is not None else None for value in column]
  File "/usr/local/lib/python3.10/dist-packages/datasets/features/features.py", line 1414, in decode_nested_example
    return schema.decode_example(obj, token_per_repo_id=token_per_repo_id) if obj is not None else None
  File "/usr/local/lib/python3.10/dist-packages/datasets/features/audio.py", line 188, in decode_example
    array = librosa.to_mono(array)
  File "/usr/local/lib/python3.10/dist-packages/lazy_loader/__init__.py", line 83, in __getattr__
    attr = getattr(submod, name)
  File "/usr/local/lib/python3.10/dist-packages/lazy_loader/__init__.py", line 82, in __getattr__
    submod = importlib.import_module(submod_path)
  File "/usr/lib/python3.10/importlib/__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 1050, in _gcd_import
  File "<frozen importlib._bootstrap>", line 1027, in _find_and_load
  File "<frozen importlib._bootstrap>", line 1006, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 688, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 883, in exec_module
  File "<frozen importlib._bootstrap>", line 241, in _call_with_frames_removed
  File "/usr/local/lib/python3.10/dist-packages/librosa/core/audio.py", line 20, in <module>
    from .convert import frames_to_samples, time_to_samples
  File "/usr/local/lib/python3.10/dist-packages/librosa/core/convert.py", line 6, in <module>
    from . import notation
  File "/usr/local/lib/python3.10/dist-packages/librosa/core/notation.py", line 1015, in <module>
    def __o_fold(d):
  File "/usr/local/lib/python3.10/dist-packages/numba/core/decorators.py", line 225, in wrapper
    disp.enable_caching()
  File "/usr/local/lib/python3.10/dist-packages/numba/core/dispatcher.py", line 807, in enable_caching
    self._cache = FunctionCache(self.py_func)
  File "/usr/local/lib/python3.10/dist-packages/numba/core/caching.py", line 647, in __init__
    self._impl = self._impl_class(py_func)
  File "/usr/local/lib/python3.10/dist-packages/numba/core/caching.py", line 383, in __init__
    raise RuntimeError("cannot cache function %r: no locator available "
RuntimeError: cannot cache function '__o_fold': no locator available for file '/usr/local/lib/python3.10/dist-packages/librosa/core/notation.py'
