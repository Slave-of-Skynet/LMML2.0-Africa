"""Denoise and transcribe the official isiZulu stadium recording locally.

Generated audio/transcripts belong outside the tracked solution directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import ndimage, signal
from scipy.io import wavfile


def read_audio(path: Path):
    rate, audio = wavfile.read(path)
    if rate != 16000 or audio.ndim != 1 or audio.dtype != np.int16:
        raise ValueError("Expected the official mono 16 kHz PCM16 input")
    return audio.astype(np.float32) / 32768


def denoise(audio):
    rate = 16000
    _, _, spectrum = signal.stft(audio, rate, nperseg=512, noverlap=384)
    magnitude = abs(spectrum)
    floor = np.percentile(magnitude, 20, axis=1, keepdims=True)
    gain = np.maximum(0.03, 1 - (floor / (magnitude + 1e-8)) ** 1.5)
    _, clean = signal.istft(spectrum * gain, rate, nperseg=512, noverlap=384)
    clean = signal.sosfilt(
        signal.butter(4, [250, 6500], btype="bandpass", fs=rate, output="sos"),
        clean,
    )
    return (clean[: len(audio)] * (0.95 / (abs(clean).max() + 1e-8))).astype(np.float32)


def remove_sustained_tones(audio):
    """Remove long horn harmonics with a 0.736-second time opening."""
    rate = 16000
    _, _, spectrum = signal.stft(audio, rate, nperseg=2048, noverlap=1536)
    magnitude = abs(spectrum)
    baseline = ndimage.maximum_filter1d(
        ndimage.minimum_filter1d(magnitude, size=23, axis=1), size=23, axis=1
    )
    gain = np.maximum(0.02, 1 - 2 * baseline / (magnitude + 1e-9))
    _, clean = signal.istft(spectrum * gain, rate, nperseg=2048, noverlap=1536)
    clean = signal.sosfilt(
        signal.butter(4, [200, 7000], btype="bandpass", fs=rate, output="sos"), clean
    )
    clean = clean[: len(audio)]
    clean = clean / (abs(clean).max() + 1e-9) * 0.95
    # Match the independently generated PCM16 analysis audio exactly.
    return (clean * 32767).astype(np.int16).astype(np.float32) / 32768


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", type=str)
    parser.add_argument("--revision", default="35d82a432d91297c4c9afc126ecac71910f25b2e")
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--end", type=float)
    parser.add_argument("--raw", action="store_true")
    parser.add_argument("--method", choices=["morph", "spectral"], default="morph")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--tokens", action="store_true", help="Include approximate CTC token times")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    audio = read_audio(args.audio)
    clean = audio if args.raw else (
        remove_sustained_tones(audio) if args.method == "morph" else denoise(audio)
    )
    wavfile.write(
        args.out / "processed.wav", 16000,
        np.clip(clean * 32768, -32768, 32767).astype(np.int16),
    )
    metadata = {
        "input_sha256": hashlib.sha256(args.audio.read_bytes()).hexdigest(),
        "sample_rate": 16000,
        "duration_seconds": len(audio) / 16000,
        "method": "raw" if args.raw else args.method,
    }
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if not args.model:
        return
    import torch
    from transformers import AutoProcessor, Wav2Vec2ForCTC

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    processor = AutoProcessor.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=False
    )
    model = Wav2Vec2ForCTC.from_pretrained(
        args.model, revision=args.revision, use_safetensors=True
    ).eval()
    limit = min(len(audio) / 16000, args.end or len(audio) / 16000)
    with (args.out / "transcript.jsonl").open("w", encoding="utf-8") as stream:
        for start in np.arange(args.start, limit, 18):
            end = min(start + 20, limit)
            chunk = clean[int(start * 16000): int(end * 16000)]
            inputs = processor(chunk, sampling_rate=16000, return_tensors="pt")
            with torch.inference_mode():
                logits = model(**inputs).logits
            token_ids = logits.argmax(dim=-1)
            text = processor.batch_decode(token_ids)[0]
            row = {"start": float(start), "end": float(end), "text": text}
            if args.tokens:
                ids = token_ids[0].tolist()
                tokens = []
                previous = None
                stride = int(np.prod(model.config.conv_stride)) / 16000
                for frame, token_id in enumerate(ids):
                    if token_id != previous and token_id != model.config.pad_token_id:
                        tokens.append({
                            "time": round(float(start) + frame * stride, 3),
                            "token": processor.tokenizer.convert_ids_to_tokens(token_id),
                        })
                    previous = token_id
                row["tokens"] = tokens
            print(json.dumps(row, ensure_ascii=False), flush=True)
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()


if __name__ == "__main__":
    main()
