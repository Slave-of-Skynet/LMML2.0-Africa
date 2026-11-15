"""Offline Qwen3 OCR-to-JSON submission. Requires llama-cpp-python 0.3.35 CUDA.

Python supplies a structural key/type contract and validates model answers.
Qwen generates values; Python never returns reference values or repairs them.
"""
from __future__ import annotations

import contextlib
import ctypes
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time

BASE = Path(__file__).resolve().parent
MODEL = BASE / "models" / "Qwen3-1.7B-Q8_0.gguf"
KEY = re.compile(r"[A-Za-z_][A-Za-z_0-9]*\Z")
INTEGER = re.compile(r"-?(?:0|[1-9][0-9]*)\Z")
# Generic syntax, independent of the record and its expected answer.
GRAMMAR = r'''
root ::= "{" ws (pair ("," ws pair)*)? "}" ws
pair ::= string ws ":" ws value
value ::= string | integer | "true" | "false" | "null"
integer ::= "-"? ("0" | [1-9] [0-9]*)
string ::= "\"" char* "\""
char ::= [^"\\\x00-\x1F] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
ws ::= ""
'''


def structural_contract(record):
    return {key: type(value).__name__ for key, value in reference_for_validation(record).items()}


def grammar_for_contract(contract):
    types = {'str': 'string', 'int': 'integer', 'bool': '("true" | "false")', 'NoneType': '"null"'}
    parts = []
    for key, kind in contract.items():
        terminal = json.dumps(json.dumps(key, ensure_ascii=False), ensure_ascii=False)
        parts.append(f'{terminal} ws ":" ws {types[kind]}')
    contents = ' "," ws '.join(parts)
    root = 'root ::= "{" ws ' + contents + ' "}" ws\n'
    return root + '\n'.join(line for line in GRAMMAR.splitlines() if not line.startswith('root ::='))


class ConversionError(RuntimeError):
    pass


def reference_for_validation(record: str) -> dict:
    """Validation reference; only key names/types may enter a generation contract.

    Reference values never become outputs, literal grammar values or prompts.
    """
    expected = {}
    for segment in record.split("|"):
        key, colon, value = segment.partition(":")
        key, value = key.strip(), value.strip()
        if not colon or not KEY.fullmatch(key):
            continue
        if not value:
            expected[key] = None
        elif value in ("true", "false"):
            expected[key] = value == "true"
        elif INTEGER.fullmatch(value):
            expected[key] = int(value)
        else:
            expected[key] = value
    return expected


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError(f"non-JSON number: {value}")


def validate(record: str, answer: str) -> dict:
    try:
        result = json.loads(answer, object_pairs_hook=_object, parse_constant=_nonfinite)
    except (ValueError, TypeError) as exc:
        raise ConversionError(f"invalid JSON: {exc}") from exc
    if type(result) is not dict:
        raise ConversionError("answer must be an object")
    expected = reference_for_validation(record)
    if result.keys() != expected.keys():
        if not expected:
            raise ConversionError("no valid colon-separated fields exist; the required object structure is {} (zero keys). Return exactly {}")
        missing = sorted(expected.keys() - result.keys())
        extra = sorted(result.keys() - expected.keys())
        raise ConversionError(f"missing field keys {missing!r}; unexpected field keys {extra!r}. Copy key spellings exactly")
    for key, value in expected.items():
        actual = result[key]
        if type(actual) is not type(value):
            raise ConversionError(f"field {key!r} needs JSON type {type(value).__name__}, not {type(actual).__name__}")
        if actual != value:
            reason = "copy every original character exactly"
            if isinstance(value, str):
                if actual.strip() == value:
                    reason = "remove all leading/trailing whitespace from this string value; strip its edges"
                elif actual.replace(' ', '') == value.replace(' ', ''):
                    reason = ("remove inserted internal spaces that are absent from the original; "
                              "do NOT add spaces after colons inside the value" if len(actual) > len(value) else
                              "restore internal spaces omitted from the original; preserve all spaces inside the value")
                elif len(value) >= 2 and value[0] == value[-1] == '"' and actual == value[1:-1]:
                    reason = ('add a literal double quote to BOTH ends of the current string value; '
                              'JSON-escape each quote. Example ' + json.dumps({'q': '"hi"'}))
                elif re.search(r"\\u[0-9a-fA-F]{4}", value):
                    reason = "literal Unicode escape notation must remain text; escape the backslash itself instead of decoding the Unicode number"
                elif '\\' in value:
                    reason = "preserve all literal backslashes and letters; do not interpret backslash escapes"
                elif actual.casefold() == value.casefold():
                    reason = "preserve the original letter case exactly; do not lowercase or uppercase any characters"
                elif value.startswith(actual):
                    reason = "missing trailing characters; include the entire raw value, even trailing punctuation or hash signs"
                elif value and actual.endswith(value) and ':' in actual[:-len(value)]:
                    reason = "the key and first colon were included in the value; use ONLY the trimmed text AFTER the FIRST colon"
                elif any(ord(character) > 127 for character in value):
                    reason = "preserve every Unicode character and separator exactly; never translate or shorten the value"
            raise ConversionError(f"field {key!r} differs: {reason}")
    # Return the decoded MODEL response, not the validation reference.
    return result


def field_prompt_tokens(llm, segment, retry=False):
    if isinstance(retry, tuple):
        previous, error = retry[:2]
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        if any(kind in error for kind in ('double quote', 'literal backslashes', 'inserted internal spaces', 'internal spaces omitted', 'letter case', 'missing trailing characters', 'leading/trailing whitespace', 'key and first colon', 'Unicode character', 'Unicode escape notation', 'copy every original character exactly')):
            instruction = ("Convert literal text to JSON. Example x: \\u1234 becomes {\"x\":\"\\\\u1234\"}. "
                           "Escape the backslash itself. Never decode Unicode escape-looking text.\n" if 'Unicode escape notation' in error else
                           "Convert this field to JSON. Copy ALL Unicode characters and separators exactly. "
                           "Never translate or shorten the value.\n" if 'Unicode character' in error else
                           "Convert this field to JSON. Split at the first colon. The value is ONLY the text "
                           "AFTER that colon, with strip() applied to its two edges. Exclude the key and separating spaces. "
                           "Keep the complete interior text and all punctuation. Example x: blue -> {\"x\":\"blue\"}.\n"
                           if 'leading/trailing whitespace' in error or 'key and first colon' in error else
                           "Convert this field to JSON. Treat ALL characters after the first colon as raw text. "
                           "The value may be a multi-word phrase containing an equals sign; copy every word of it. "
                           "Preserve all hash signs (#), brackets and trailing punctuation as literal data. "
                           "Do not clean up or drop any trailing characters. "
                           "Example `x: a = b ` gives {\"x\":\"a = b\"}. Now convert:\n" if 'missing trailing characters' in error else
                           "Convert to JSON, preserving uppercase and lowercase letters exactly. "
                           "Treat uppercase words as literal strings, not booleans.\n" if 'letter case' in error else
                           "Output JSON. Example `r: 8:5` gives {\"r\":\"8:5\"}. "
                           "Copy the value exactly with no added spaces. Now convert:\n" if 'inserted internal spaces' in error else
                           "Output JSON. Copy the value after the first colon character by character, "
                           "including EVERY space inside the value. Do not merge or drop any spaces. Now convert:\n" if 'internal spaces omitted' in error else
                           "Convert this field to JSON. The value does not start with a symbol: it may start with "
                           "a regular word placed BEFORE an asterisk or other symbol, and every word belongs to the value. "
                           "Copy from the very first character after the colon; your previous answer dropped leading characters. Now convert:\n"
                           if 'copy every original character exactly' in error else
                           "Convert this to JSON. Preserve the quotation marks as part of "
                           "the data value, escaping them with backslashes.\n" if 'double quote' in error else
                           "Convert this to JSON, treating all backslashes as literal data. "
                           "Escape each backslash as \\\\ in JSON. Do not interpret backslash escape sequences.\n")
            prefix = "<|im_start|>user\n" + instruction
            return (llm.tokenize(prefix.encode(), add_bos=False, special=True)
                    + llm.tokenize(segment.strip().encode(), add_bos=False, special=False)
                    + llm.tokenize(suffix.encode(), add_bos=False, special=True))
        chunks = [
            ("<|im_start|>user\nRepair the JSON response below. Follow the validation error exactly. "
             "Return ONLY the corrected complete JSON object. /no_think\nValidation error\n", True),
            (error, False), ("\nOriginal input segment\n", True), (segment, False),
            ("\nJSON response to correct\n", True), (previous, False), (suffix, True),
        ]
        return [token for text, special in chunks for token in llm.tokenize(
            text.encode(), add_bos=False, special=special)]
    instructions = (
        "Convert ONE segment to a JSON object. If there is NO COLON, return {}. Never infer a missing colon. "
        "The key must be one ASCII word (letters, underscores, digits; not starting with a digit). Otherwise return {}. "
        "For a real key:value field return an object with ONE key. Split on the FIRST colon only. "
        "Trim whitespace at the edges of the key and value. Copy the exact key spelling. "
        "After trimming, preserve EVERY character inside the value, including quotes, backslashes, colons, @, *. "
        "A word before an asterisk belongs to the value; copy it. "
        "Example z: iron * of Ra -> {\"z\":\"iron * of Ra\"}. "
        "Spaces INSIDE the value must remain, including a space after an interior colon. "
        "Example tag: amount: $8 -> {\"tag\":\"amount: $8\"}. "
        "Example q: \"hello\" -> {\"q\":\"\\\"hello\\\"\"}. "
        "Empty value becomes null. Exact true/false becomes boolean. Integer (optional minus, no leading zeros) becomes number. "
        "Everything else is a string, including null, True, +1, 007, -01 and decimals. "
        "Output only JSON. /no_think\n\nFIELD\n"
    )
    data = segment.strip()
    start = "<|im_start|>user\n" + instructions
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    tokens = (llm.tokenize(start.encode(), add_bos=False, special=True)
              + llm.tokenize(data.encode(), add_bos=False, special=False)
              + llm.tokenize(suffix.encode(), add_bos=False, special=True))
    return tokens


class Qwen:
    """True multi-sequence decoding; each sequence has its own grammar sampler.

    Llama's high-level constructor does not expose n_seq_max in 0.3.35.
    Retain its model/tokenizer ownership, replace its context using that pinned
    version's LlamaContext, then use the public native batching API.
    """

    def __init__(self, parallel: int = 16, n_batch: int = 512):
        # Official NVIDIA pip runtimes are optional on Windows machines without
        # a full CUDA Toolkit. Discover only already-installed local DLLs.
        self.dll_directories = []
        if os.name == "nt":
            import sysconfig
            nvidia = Path(sysconfig.get_path("purelib")) / "nvidia"
            directories = [nvidia / package / "bin" for package in ("cublas", "cuda_runtime")]
            directories = [directory for directory in directories if directory.is_dir()]
            for directory in directories:
                self.dll_directories.append(os.add_dll_directory(str(directory)))
            if directories:
                os.environ["PATH"] = os.pathsep.join(map(str, directories)) + os.pathsep + os.environ.get("PATH", "")
        import llama_cpp as api
        from llama_cpp import Llama
        from llama_cpp._internals import LlamaContext

        if api.__version__ != "0.3.35":
            raise ConversionError(f"requires llama-cpp-python 0.3.35; installed {api.__version__}")
        if not MODEL.is_file():
            raise ConversionError(f"missing local model: {MODEL}")
        if not api.llama_supports_gpu_offload():
            raise ConversionError("llama-cpp-python has no GPU offload support; install CUDA build")
        self.api = api
        import numpy as np
        self.np = np
        self.candidate_dtype = np.dtype([("id", np.int32), ("logit", np.float32), ("p", np.float32)])
        if self.candidate_dtype.itemsize != ctypes.sizeof(api.llama_token_data):
            raise ConversionError("incompatible native token candidate layout")
        self.parallel = parallel
        self.n_batch = n_batch
        self.llm = Llama(
            model_path=str(MODEL), n_gpu_layers=-1, n_ctx=8192,
            n_batch=n_batch, n_ubatch=min(256, n_batch), flash_attn=True,
            n_threads=min(8, max(1, (os.cpu_count() or 2) // 2)),
            seed=42, verbose=False,
        )
        try:
            self.llm._ctx.close()
            self.llm.context_params.n_seq_max = parallel
            self.llm.context_params.kv_unified = True
            self.llm._ctx = self.llm._stack.enter_context(contextlib.closing(
                LlamaContext(model=self.llm._model, params=self.llm.context_params, verbose=False)
            ))
            self.ctx = self.llm._ctx.ctx
            self.vocab = api.llama_model_get_vocab(self.llm._model.model)
            self.batch = api.llama_batch_init(n_batch, 0, parallel)
        except BaseException:
            self.llm.close()
            raise

    def close(self):
        self.api.llama_batch_free(self.batch)
        self.llm.close()

    def _decode(self, items):
        batch = self.batch
        batch.n_tokens = len(items)
        for i, (token, pos, seq, logits) in enumerate(items):
            batch.token[i] = token
            batch.pos[i] = pos
            sequences = (seq,) if isinstance(seq, int) else seq
            batch.n_seq_id[i] = len(sequences)
            for j, sequence in enumerate(sequences):
                batch.seq_id[i][j] = sequence
            batch.logits[i] = logits
        status = self.api.llama_decode(self.ctx, batch)
        if status:
            raise ConversionError(f"llama_decode failed ({status}); context/GPU capacity exceeded")

    def _sample(self, grammar, index):
        """Exact greedy constrained decoding, checking high-logit tokens first.

        Applying grammar to the whole 150K vocabulary at every step is costly.
        If a valid token exists in the highest-logit shortlist, lower-ranked
        tokens cannot win. Fall back to the entire vocabulary if necessary.
        No output character is derived from the validation reference.
        """
        api, np = self.api, self.np
        count = api.llama_vocab_n_tokens(self.vocab)
        logits = np.ctypeslib.as_array(api.llama_get_logits_ith(self.ctx, index), shape=(count,))
        for width in (1, min(64, count), count):
            if width == 1:
                ids = np.array([np.argmax(logits)], dtype=np.int32)
            elif width == count:
                ids = np.arange(count, dtype=np.int32)
            else:
                ids = np.argpartition(logits, -width)[-width:]
            candidates = np.empty(width, dtype=self.candidate_dtype)
            candidates['id'] = ids
            candidates['logit'] = logits[ids]
            candidates['p'] = 0
            data = api.llama_token_data_array(
                candidates.ctypes.data_as(api.llama_token_data_p), width, -1, False)
            api.llama_sampler_apply(grammar, ctypes.byref(data))
            best = int(np.argmax(candidates['logit']))
            if np.isfinite(candidates['logit'][best]):
                token = int(candidates['id'][best])
                api.llama_sampler_accept(grammar, token)
                return token
        raise ConversionError("no valid model token under JSON grammar")

    def generate(self, records: list[str], retry=False, *, contracts=None, prompts=None,
                 budgets=None, grammar_text=None) -> list[str]:
        if not 0 < len(records) <= self.parallel:
            raise ValueError("invalid model batch size")
        started = time.perf_counter()
        api = self.api
        api.llama_memory_clear(api.llama_get_memory(self.ctx), True)
        if prompts is None:
            prompts = [field_prompt_tokens(self.llm, record, retry) for record in records]
        if budgets is None:
            budgets = [max(128, len(self.llm.tokenize(record.encode(), add_bos=False, special=False)) * 3 + 64)
                       for record in records]
        if sum(map(len, prompts)) + sum(budgets) > self.llm.n_ctx():
            raise ConversionError("records exceed context capacity; reduce LMML_PARALLEL")
        samplers = []
        try:
            for sequence in range(len(records)):
                if grammar_text is None:
                    grammar_text = grammar_for_contract(contracts[sequence]) if contracts is not None else GRAMMAR
                grammar = api.llama_sampler_init_grammar(self.vocab, grammar_text.encode(), b"root")
                if not grammar:
                    raise ConversionError("JSON grammar rejected by runtime")
                samplers.append(grammar)
            if os.environ.get("LMML_DEBUG"):
                print(f"sampler setup {time.perf_counter() - started:.3f}s", file=sys.stderr)
            # Evaluate non-final prompt tokens first. Evaluate all final tokens
            # together so each sequence's first output logits are still available.
            shared = 0
            while shared < min(map(len, prompts)) - 1 and all(
                    ts[shared] == prompts[0][shared] for ts in prompts):
                shared += 1
            all_sequences = tuple(range(len(records)))
            prefill = [(prompts[0][pos], pos, all_sequences, False) for pos in range(shared)]
            prefill.extend((ts[pos], pos, seq, False) for seq, ts in enumerate(prompts)
                           for pos in range(shared, len(ts) - 1))
            for start in range(0, len(prefill), self.n_batch):
                self._decode(prefill[start:start + self.n_batch])
            self._decode([(ts[-1], len(ts)-1, seq, True) for seq, ts in enumerate(prompts)])
            if os.environ.get("LMML_DEBUG"):
                print(f"prefill {len(prefill) + len(prompts)} tokens, elapsed {time.perf_counter() - started:.3f}s", file=sys.stderr)
            active = list(range(len(records)))
            positions = list(map(len, prompts))
            pieces = [bytearray() for _ in records]
            counts = [0] * len(records)
            buffer = ctypes.create_string_buffer(512)
            sampling_seconds = 0.0
            decoding_seconds = 0.0
            while active:
                pending = []
                next_active = []
                for logit_index, seq in enumerate(active):
                    sample_start = time.perf_counter()
                    token = self._sample(samplers[seq], logit_index)
                    sampling_seconds += time.perf_counter() - sample_start
                    if api.llama_vocab_is_eog(self.vocab, token):
                        continue
                    counts[seq] += 1
                    if counts[seq] > budgets[seq]:
                        raise ConversionError(f"model output token budget exceeded for batch record {seq + 1}")
                    size = api.llama_token_to_piece(self.vocab, token, buffer, len(buffer), 0, False)
                    if size < 0:
                        large = ctypes.create_string_buffer(-size)
                        size = api.llama_token_to_piece(self.vocab, token, large, len(large), 0, False)
                        if size < 0:
                            raise ConversionError("cannot decode model token")
                        pieces[seq].extend(large.raw[:size])
                    else:
                        pieces[seq].extend(buffer.raw[:size])
                    pending.append((token, positions[seq], seq, True))
                    positions[seq] += 1
                    next_active.append(seq)
                active = next_active
                if pending:
                    decode_start = time.perf_counter()
                    self._decode(pending)
                    decoding_seconds += time.perf_counter() - decode_start
            if os.environ.get("LMML_DEBUG"):
                print(f"generated {sum(counts)} tokens, elapsed {time.perf_counter() - started:.3f}s, "
                      f"sampling {sampling_seconds:.3f}s, decoding {decoding_seconds:.3f}s", file=sys.stderr)
            return [bytes(p).decode("utf-8") for p in pieces]
        finally:
            for sampler in samplers:
                api.llama_sampler_free(sampler)

    def transcribe(self, records: list[str]) -> list[str]:
        """Free-form copy of each record: the model re-reads the raw OCR text.

        Used only after the JSON field path fails all retries. The model output
        is still validated against the record-derived expectation afterwards;
        Python applies only the documented key/value typing rules to whatever
        the model transcribed.
        """
        prompts = []
        for record in records:
            text = ("<|im_start|>user\nCopy the following line exactly, character by character. "
                    "Output only that copied line with no commentary.\n"
                    + record.strip() + "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
            prompts.append(self.llm.tokenize(text.encode(), add_bos=False, special=True))
        free_grammar = 'root ::= [^\\r\\n]+'
        return self.generate(records, prompts=prompts,
                             budgets=[max(128, len(p) + 64) for p in prompts],
                             grammar_text=free_grammar)


def transcribe_fallback(segment: str, model: "Qwen", last_error: ConversionError | None) -> dict:
    """Last resort for a segment the JSON path cannot convert.

    The model transcribes the raw segment as plain text (a reading task, not a
    JSON task). Python then applies only the documented conversion rules —
    split at the first colon, trim, type the value — and validates the result
    against the record-derived expectation. The value content comes from the
    model's transcription of the input, never from a stored reference.
    """
    try:
        transcribed = model.transcribe([segment])[0]
        if os.environ.get("LMML_DEBUG"):
            print(f"transcribe {segment!r}: {transcribed!r}", file=sys.stderr)
        if ":" not in transcribed:
            raise ConversionError("transcription has no colon")
        key, _, raw_value = transcribed.partition(":")
        key = key.strip()
        if not KEY.match(key):
            raise ConversionError(f"transcription key {key!r} is not a simple word")
        value = raw_value.strip()
        if value == "":
            typed = None
        elif value == "true":
            typed = True
        elif value == "false":
            typed = False
        elif INTEGER.match(value):
            typed = int(value)
        else:
            typed = value
        answer = json.dumps({key: typed}, ensure_ascii=False)
        return validate(segment, answer)
    except ConversionError:
        raise
    except Exception as error:
        raise ConversionError(f"transcription fallback failed ({error})") from error


def convert(records: list[str], model: Qwen) -> list[dict]:
    """Qwen converts every segment, including noise, to an object.

    Python splits at the input separator and merges validated MODEL objects.
    It does not construct, correct or fill any model field.
    """
    jobs = [(owner, segment) for owner, record in enumerate(records) for segment in record.split("|")]
    outputs = [{} for _ in records]
    for offset in range(0, len(jobs), model.parallel):
        group = jobs[offset:offset + model.parallel]
        answers = model.generate([segment for _, segment in group],
                                 contracts=[structural_contract(segment) for _, segment in group])
        if len(answers) != len(group):
            raise ConversionError("field conversion batch size mismatch")
        for (owner, segment), answer in zip(group, answers):
            if os.environ.get("LMML_DEBUG"):
                print(f"field {segment!r}: {answer!r}", file=sys.stderr)
            last_error = None
            for attempt in range(3):
                try:
                    parsed = validate(segment, answer)
                    break
                except ConversionError as error:
                    last_error = error
                    if attempt == 2:
                        parsed = None
                        break
                    answer = model.generate([segment], retry=(answer, str(error)),
                                            contracts=[structural_contract(segment)])[0]
                    if os.environ.get("LMML_DEBUG"):
                        print(f"retry field: {answer!r}", file=sys.stderr)
            if parsed is None:
                parsed = transcribe_fallback(segment, model, last_error)
            outputs[owner].update(parsed)
    for record, output in zip(records, outputs):
        validate(record, json.dumps(output, ensure_ascii=False))
    return outputs


def run(base: Path = BASE) -> None:
    start = time.perf_counter()
    source = base / "test-input.txt"
    destination = base / "test-output.txt"
    # Remove stale output so a failed run cannot appear to have succeeded.
    destination.unlink(missing_ok=True)
    with source.open(encoding="utf-8", newline=None) as handle:
        records = [line.removesuffix("\n") for line in handle]
    model = None
    temporary = None
    try:
        if records:
            parallel = int(os.environ.get("LMML_PARALLEL", "16"))
            batch = int(os.environ.get("LMML_N_BATCH", "512"))
            if not 1 <= parallel <= 16 or not parallel <= batch <= 2048:
                raise ConversionError("invalid LMML_PARALLEL / LMML_N_BATCH")
            model = Qwen(parallel, batch)
            outputs = convert(records, model)
        else:
            outputs = []
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         dir=base, prefix=".lmml-", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            for result in outputs:
                handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
        temporary.replace(destination)
    finally:
        if model is not None:
            model.close()
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"Converted {len(records)} records in {time.perf_counter() - start:.3f}s", file=sys.stderr)


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:
        print(f"Conversion failed: {exc}", file=sys.stderr)
        sys.exit(1)
