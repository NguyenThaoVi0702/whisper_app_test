I was testing out the new model using this code:
"import os
import logging
from typing import List, Dict, Any, Tuple

from faster_whisper import WhisperModel
from app.core.config import settings

logger = logging.getLogger(__name__)

class Transcriber:

    def __init__(self, model_path: str = settings.FASTER_WHISPER_MODEL_PATH):
        """
        Initializes the Transcriber and loads the Faster-Whisper model into memory.
        """
        if not model_path or not os.path.isdir(model_path):
            msg = f"Faster-Whisper model path '{model_path}' is invalid or does not exist."
            logger.critical(msg)
            raise FileNotFoundError(msg)

        logger.info(f"Loading Faster-Whisper model from: {model_path}")
        try:
            self.model = WhisperModel(
                model_path,
                device="cuda",
                compute_type="float16",
                local_files_only=True
            )
            logger.info("Faster-Whisper model loaded successfully onto CUDA device.")
        except Exception as e:
            logger.critical(f"FATAL: Failed to load Faster-Whisper model: {e}", exc_info=True)
            raise

    def transcribe(self, audio_path: str, language: str = "vi") -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Transcribes an audio file and produces two separate transcript formats.

        1.  Sentence-level transcript: Logical chunks of speech suitable for display.
        2.  Word-level transcript: Granular word data with precise timestamps for mapping.
        """
        logger.info(f"Starting transcription for audio file: {audio_path} in language '{language}'")

        try:

            segments, _ = self.model.transcribe(
                audio_path,
                beam_size=5,
                language=language,
                word_timestamps=True,  
                vad_filter=False,
                vad_parameters=dict(min_silence_duration_ms=500),
            )

            sentence_level_transcript = []
            word_level_transcript = []
            segment_id_counter = 0

            for segment in segments:
                # --- 1. Build the Sentence-Level Transcript (for the user) ---
                sentence_level_transcript.append({
                    "id": segment_id_counter,
                    "text": segment.text.strip(),
                    "start_time": round(segment.start, 3),
                    "end_time": round(segment.end, 3),
                })
                segment_id_counter += 1

                # --- 2. Build the Word-Level Transcript (for the database/diarization) ---
                if segment.words:
                    for word in segment.words:
                        word_level_transcript.append({
                            "word": word.word.strip(),
                            "start": round(word.start, 3),
                            "end": round(word.end, 3),
                        })

            logger.info(f"Successfully transcribed {audio_path}. "
                        f"Found {len(sentence_level_transcript)} sentences and {len(word_level_transcript)} words.")

            return sentence_level_transcript, word_level_transcript

        except Exception as e:
            logger.error(f"An unexpected error occurred during transcription of {audio_path}: {e}", exc_info=True)
            return [], []"

but got this error, remember that my company PC cannot connect to the internet:
"eeting_worker      |       warnings.warn(SecurityWarning(ROOT_DISCOURAGED.format(
meeting_worker      |
meeting_worker      | [2025-10-17 10:32:14,367: INFO/MainProcess] Loading Faster-Whisper model from: /code/models/new_whisper_vietbud500_ct2_model
meeting_worker      | [2025-10-17 10:32:31,631: CRITICAL/MainProcess] FATAL: Failed to load Faster-Whisper model: An error happened while trying to locate the file on the Hub and we cannot find the requested files in the local cache. Please check your connection and try again or make sure your Internet connection is on.
meeting_worker      | Traceback (most recent call last):
meeting_worker      |   File "/usr/local/lib/python3.12/dist-packages/urllib3/connection.py", line 198, in _new_conn
meeting_worker      |     sock = connection.create_connection(
meeting_worker      |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
meeting_worker      |   File "/usr/local/lib/python3.12/dist-packages/urllib3/util/connection.py", line 60, in create_connection
meeting_worker      |     for res in socket.getaddrinfo(host, port, family, socket.SOCK_STREAM):
meeting_worker      |                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
meeting_worker      |   File "/usr/lib/python3.12/socket.py", line 978, in getaddrinfo
meeting_worker      |     for res in _socket.getaddrinfo(host, port, family, type, proto, flags):
meeting_worker      |                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
meeting_worker      | socket.gaierror: [Errno -3] Temporary failure in name resolution
meeting_worker      |
meeting_worker      | The above exception was the direct cause of the following exception:
meeting_worker      |
meeting_worker      | Traceback (most recent call last):
meeting_worker      |   File "/usr/local/lib/python3.12/dist-packages/urllib3/connectionpool.py", line 787, in urlopen
meeting_worker      |     response = self._make_request(
meeting_worker      |                ^^^^^^^^^^^^^^^^^^^
meeting_worker      |   File "/usr/local/lib/python3.12/dist-packages/urllib3/connectionpool.py", line 488, in _make_request
meeting_worker      |     raise new_e
meeting_worker      |   File "/usr/local/lib/python3.12/dist-packages/urllib3/connectionpool.py", line 464, in _make_request
meeting_worker      |     self._validate_conn(conn)
meeting_worker      |   File "/usr/local/lib/python3.12/dist-packages/urllib3/connectionpool.py", line 1093, in _validate_conn
meeting_worker      |     conn.connect()
meeting_worker      |   File "/usr/local/lib/python3.12/dist-packages/urllib3/connection.py", line 704, in connect
meeting_worker      |     self.sock = sock = self._new_conn()
meeting_worker      |                        ^^^^^^^^^^^^^^^^
meeting_worker      |   File "/usr/local/lib/python3.12/dist-packages/urllib3/connection.py", line 205, in _new_conn
meeting_worker      |     raise NameResolutionError(self.host, self, e) from e
meeting_worker      | urllib3.exceptions.NameResolutionError: <urllib3.connection.HTTPSConnection object at 0x7fd3a00cc080>: Failed to resolve 'huggingface.co' ([Errno -3] Temporary failure in name resolution)
meeting_worker      |
meeting_worker      | The above exception was the direct cause of the following exception:
meeting_worker      |
meeting_worker      | Traceback (most recent call last):
meeting_worker      |   File "/usr/local/lib/python3.12/dist-packages/requests/adapters.py", line 667, in send
meeting_worker      |     resp = conn.urlopen(
meeting_worker      |            ^^^^^^^^^^^^^
meeting_worker      |   File "/usr/local/lib/python3.12/dist-packages/urllib3/connectionpool.py", line 841, in urlopen
meeting_worker      |     retries = retries.increment(
meeting_worker      |               ^^^^^^^^^^^^^^^^^^
meeting_worker      |   File "/usr/local/lib/python3.12/dist-packages/urllib3/util/retry.py", line 519, in increment
meeting_worker      |     raise MaxRetryError(_pool, url, reason) from reason  # type: ignore[arg-type]
meeting_worker      |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
meeting_worker      | urllib3.exceptions.MaxRetryError: HTTPSConnectionPool(host='huggingface.co', port=443): Max retries exceeded with url: /openai/whisper-tiny/resolve/main/tokenizer.json (Caused by NameResolutionError("<urllib3.connection.HTTPSConnection object at 0x7fd3a00cc080>: Failed to resolve 'huggingface.co' ([Errno -3] Temporary failure in name resolution)"))
meeting_worker      |
meeting_worker      | During handling of the above exception, another exception occurred:
meeting_worker      |
meeting_worker      | Traceback (most recent call last):
meeting_worker      |   File "/usr/local/lib/python3.12/dist-packages/huggingface_hub/file_download.py", line 1484, in _get_metadata_or_catch_error
meeting_worker      |     metadata = get_hf_file_metadata(
meeting_worker      |                ^^^^^^^^^^^^^^^^^^^^^
meeting_worker      |   File "/usr/local/lib/python3.12/dist-packages/huggingface_hub/utils/_validators.py", line 114, in _inner_fn
meeting_worker      |     return fn(*args, **kwargs)
meeting_worker      |            ^^^^^^^^^^^^^^^^^^^
meeting_worker      |   File "/usr/local/lib/python3.12/dist-packages/huggingface_hub/file_download.py", line 1401, in get_hf_file_metadata
meeting_worker      |     r = _request_wrapper(
meeting_worker      |         ^^^^^^^^^^^^^^^^^
meeting_worker      |   File "/usr/local/lib/python3.12/dist-packages/huggingface_hub/file_download.py", line 285, in _request_wrapper
meeting_worker      |     response = _request_wrapper(
meeting_worker      |                ^^^^^^^^^^^^^^^^^
meeting_worker      |   File "/usr/local/lib/python3.12/dist-packages/huggingface_hub/file_download.py", line 308, in _request_wrapper
meeting_worker      |     response = get_session().request(method=method, url=url, **params)
meeting_worker      |                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
meeting_worker      |   File "/usr/local/lib/python3.12/dist-packages/requests/sessions.py", line 589, in request
meeting_worker      |     resp = self.send(prep, **send_kwargs)
meeting_worker      |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
meeting_worker      |   File "/usr/local/lib/python3.12/dist-packages/requests/sessions.py", line 703, in send
meeting_worker      |     r = adapter.send(request, **kwargs)
meeting_worker      |         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
meeting_worker      |   File "/usr/local/lib/python3.12/dist-packages/huggingface_hub/utils/_http.py", line 96, in send
meeting_worker      |     return super().send(request, *args, **kwargs)
meeting_worker      |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
meeting_worker      |   File "/usr/local/lib/python3.12/dist-packages/requests/adapters.py", line 700, in send
meeting_worker      |     raise ConnectionError(e, request=request)
meeting_worker      | requests.exceptions.ConnectionError: (MaxRetryError('HTTPSConnectionPool(host=\'huggingface.co\', port=443): Max retries exceeded with url: /openai/whisper-tiny/resolve/main/tokenizer.json (Caused by NameResolutionError("<urllib3.connection.HTTPSConnection object at 0x7fd3a00cc080>: Failed to resolve \'huggingface.co\' ([Errno -3] Temporary failure in name resolution)"))'), '(Request ID: 56b4fa30-6adc-41a6-8c6b-12ee0c2f9e35)')
meeting_worker      |
meeting_worker      | The above exception was the direct cause of the following exception:
meeting_worker      |
meeting_worker      | Traceback (most recent call last):
meeting_worker      |   File "/code/app/processing/transcription.py", line 23, in __init__
meeting_worker      |     self.model = WhisperModel(
meeting_worker      |                  ^^^^^^^^^^^^^
meeting_worker      |   File "/usr/local/lib/python3.12/dist-packages/faster_whisper/transcribe.py", line 664, in __init__
meeting_worker      |     self.hf_tokenizer = tokenizers.Tokenizer.from_pretrained(
meeting_worker      |                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
meeting_worker      |   File "/usr/local/lib/python3.12/dist-packages/huggingface_hub/utils/_validators.py", line 114, in _inner_fn
meeting_worker      |     return fn(*args, **kwargs)
meeting_worker      |            ^^^^^^^^^^^^^^^^^^^
meeting_worker      |   File "/usr/local/lib/python3.12/dist-packages/huggingface_hub/file_download.py", line 961, in hf_hub_download
meeting_worker      |     return _hf_hub_download_to_cache_dir(
meeting_worker      |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
meeting_worker      |   File "/usr/local/lib/python3.12/dist-packages/huggingface_hub/file_download.py", line 1068, in _hf_hub_download_to_cache_dir
meeting_worker      |     _raise_on_head_call_error(head_call_error, force_download, local_files_only)
meeting_worker      |   File "/usr/local/lib/python3.12/dist-packages/huggingface_hub/file_download.py", line 1599, in _raise_on_head_call_error
meeting_worker      |     raise LocalEntryNotFoundError(
meeting_worker      | huggingface_hub.errors.LocalEntryNotFoundError: An error happened while trying to locate the file on the Hub and we cannot find the requested files in the local cache. Please check your connection and try again or make sure your Internet connection is on.
meeting_worker      | [2025-10-17 10:32:31,637: ERROR/MainProcess] Signal handler <function on_worker_init at 0x7fd42215a8e0> raised: LocalEntryNotFoundError('An error happened while trying to locate the file on the Hub and we cannot find the requested files in the local cache. Please check your connection and try again or make sure your Internet connection is on.')
meeting_worker      | Traceback (most recent call last):
meeting_worker      |   File "/usr/local/lib/python3.12/dist-packages/urllib3/connection.py", line 198, in _new_conn
meeting_worker      |     sock = connection.create_connection(
meeting_worker      |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
meeting_worker      |   File "/usr/local/lib/python3.12/dist-packages/urllib3/util/connection.py", line 60, in create_connection
meeting_worker      |     for res in socket.getaddrinfo(host, port, family, socket.SOCK_STREAM):
meeting_worker      |                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
meeting_worker      |   File "/usr/lib/python3.12/socket.py", line 978, in getaddrinfo
meeting_worker      |     for res in _socket.getaddrinfo(host, port, family, type, proto, flags):
meeting_worker      |                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
meeting_worker      | socket.gaierror: [Errno -3] Temporary failure in name resolution
meeting_worker      |
meeting_worker      | The above exception was the direct cause of the following exception:
meeting_worker      |
meeting_worker      | Traceback (most recent call last):
meeting_worker      |   File "/usr/local/lib/python3.12/dist-packages/urllib3/connectionpool.py", line 787, in urlopen
meeting_worker      |     response = self._make_request(
meeting_worker      |                ^^^^^^^^^^^^^^^^^^^
meeting_worker      |   File "/usr/local/lib/python3.12/dist-packages/urllib3/connectionpool.py", line 488, in _make_request
meeting_worker      |     raise new_e
meeting_worker      |   File "/usr/local/lib/python3.12/dist-packages/urllib3/connectionpool.py", line 464, in _make_request
meeting_worker      |     self._validate_conn(conn)
meeting_worker      |   File "/usr/local/lib/python3.12/dist-packages/urllib3/connectionpool.py", line 1093, in _validate_conn
meeting_worker      |     conn.connect()
meeting_worker      |   File "/usr/local/lib/python3.12/dist-packages/urllib3/connection.py", line 704, in connect
meeting_worker      |     self.sock = sock = self._new_conn()
meeting_worker      |                        ^^^^^^^^^^^^^^^^
meeting_worker      |   File "/usr/local/lib/python3.12/dist-packages/urllib3/connection.py", line 205, in _new_conn
meeting_worker      |     raise NameResolutionError(self.host, self, e) from e
meeting_worker      | urllib3.exceptions.NameResolutionError: <urllib3.connection.HTTPSConnection object at 0x7fd3a00cc080>: Failed to resolve 'huggingface.co' ([Errno -3] Temporary failure in name resolution)
meeting_worker      |
meeting_worker      | The above exception was the direct cause of the following exception:
meeting_worker      |
meeting_worker      | Traceback (most recent call last):
meeting_worker      |   File "/usr/local/lib/python3.12/dist-packages/requests/adapters.py", line 667, in send
meeting_worker      |     resp = conn.urlopen(
meeting_worker      |            ^^^^^^^^^^^^^
meeting_worker      |   File "/usr/local/lib/python3.12/dist-packages/urllib3/connectionpool.py", line 841, in urlopen
meeting_worker      |     retries = retries.increment(
meeting_worker      |               ^^^^^^^^^^^^^^^^^^
meeting_worker      |   File "/usr/local/lib/python3.12/dist-packages/urllib3/util/retry.py", line 519, in increment
meeting_worker      |     raise MaxRetryError(_pool, url, reason) from reason  # type: ignore[arg-type]
meeting_worker      |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
meeting_worker      | urllib3.exceptions.MaxRetryError: HTTPSConnectionPool(host='huggingface.co', port=443): Max retries exceeded with url: /openai/whisper-tiny/resolve/main/tokenizer.json (Caused by NameResolutionError("<urllib3.connection.HTTPSConnection object at 0x7fd3a00cc080>: Failed to resolve 'huggingface.co' ([Errno -3] Temporary failure in name resolution)"))
meeting_worker      |
meeting_worker      | During handling of the above exception, another exception occurred:
meeting_worker      |
meeting_worker      | Traceback (most recent call last):
meeting_worker      |   File "/usr/local/lib/python3.12/dist-packages/huggingface_hub/file_download.py", line 1484, in _get_metadata_or_catch_error
meeting_worker      |     metadata = get_hf_file_metadata(
meeting_worker      |                ^^^^^^^^^^^^^^^^^^^^^
meeting_worker      |   File "/usr/local/lib/python3.12/dist-packages/huggingface_hub/utils/_validators.py", line 114, in _inner_fn
meeting_worker      |     return fn(*args, **kwargs)
meeting_worker      |            ^^^^^^^^^^^^^^^^^^^
meeting_worker      |   File "/usr/local/lib/python3.12/dist-packages/huggingface_hub/file_download.py", line 1401, in get_hf_file_metadata
meeting_worker      |     r = _request_wrapper(
meeting_worker      |         ^^^^^^^^^^^^^^^^^
meeting_worker      |   File "/usr/local/lib/python3.12/dist-packages/huggingface_hub/file_download.py", line 285, in _request_wrapper
meeting_worker      |     response = _request_wrapper(
meeting_worker      |                ^^^^^^^^^^^^^^^^^
meeting_worker      |   File "/usr/local/lib/python3.12/dist-packages/huggingface_hub/file_download.py", line 308, in _request_wrapper
meeting_worker      |     response = get_session().request(method=method, url=url, **params)
meeting_worker      |                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
meeting_worker      |   File "/usr/local/lib/python3.12/dist-packages/requests/sessions.py", line 589, in request
meeting_worker      |     resp = self.send(prep, **send_kwargs)
meeting_worker      |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
meeting_worker      |   File "/usr/local/lib/python3.12/dist-packages/requests/sessions.py", line 703, in send
meeting_worker      |     r = adapter.send(request, **kwargs)
meeting_worker      |         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
meeting_worker      |   File "/usr/local/lib/python3.12/dist-packages/huggingface_hub/utils/_http.py", line 96, in send
meeting_worker      |     return super().send(request, *args, **kwargs)
meeting_worker      |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
meeting_worker      |   File "/usr/local/lib/python3.12/dist-packages/requests/adapters.py", line 700, in send
meeting_worker      |     raise ConnectionError(e, request=request)
meeting_worker      | requests.exceptions.ConnectionError: (MaxRetryError('HTTPSConnectionPool(host=\'huggingface.co\', port=443): Max retries exceeded with url: /openai/whisper-tiny/resolve/main/tokenizer.json (Caused by NameResolutionError("<urllib3.connection.HTTPSConnection object at 0x7fd3a00cc080>: Failed to resolve \'huggingface.co\' ([Errno -3] Temporary failure in name resolution)"))'), '(Request ID: 56b4fa30-6adc-41a6-8c6b-12ee0c2f9e35)')
meeting_worker      |
meeting_worker      | The above exception was the direct cause of the following exception:
meeting_worker      |
meeting_worker      | Traceback (most recent call last):
meeting_worker      |   File "/usr/local/lib/python3.12/dist-packages/celery/utils/dispatch/signal.py", line 280, in send
meeting_worker      |     response = receiver(signal=self, sender=sender, **named)
meeting_worker      |                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
meeting_worker      |   File "/code/app/worker/tasks.py", line 34, in on_worker_init
meeting_worker      |     _transcriber_service = Transcriber()
meeting_worker      |                            ^^^^^^^^^^^^^
meeting_worker      |   File "/code/app/processing/transcription.py", line 23, in __init__
meeting_worker      |     self.model = WhisperModel(
meeting_worker      |                  ^^^^^^^^^^^^^
meeting_worker      |   File "/usr/local/lib/python3.12/dist-packages/faster_whisper/transcribe.py", line 664, in __init__
meeting_worker      |     self.hf_tokenizer = tokenizers.Tokenizer.from_pretrained(
meeting_worker      |                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
meeting_worker      |   File "/usr/local/lib/python3.12/dist-packages/huggingface_hub/utils/_validators.py", line 114, in _inner_fn
meeting_worker      |     return fn(*args, **kwargs)
meeting_worker      |            ^^^^^^^^^^^^^^^^^^^
meeting_worker      |   File "/usr/local/lib/python3.12/dist-packages/huggingface_hub/file_download.py", line 961, in hf_hub_download
meeting_worker      |     return _hf_hub_download_to_cache_dir(
meeting_worker      |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
meeting_worker      |   File "/usr/local/lib/python3.12/dist-packages/huggingface_hub/file_download.py", line 1068, in _hf_hub_download_to_cache_dir
meeting_worker      |     _raise_on_head_call_error(head_call_error, force_download, local_files_only)
meeting_worker      |   File "/usr/local/lib/python3.12/dist-packages/huggingface_hub/file_download.py", line 1599, in _raise_on_head_call_error
meeting_worker      |     raise LocalEntryNotFoundError(
meeting_worker      | huggingface_hub.errors.LocalEntryNotFoundError: An error happened while trying to locate the file on the Hub and we cannot find the requested files in the local cache. Please check your connection and try again or make sure your Internet connection is on.
"

Please help me fix the code
      
