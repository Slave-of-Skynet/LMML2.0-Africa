# Egypt — Always Valid, No Excuses

Use a local quantized model with structured decoding, targeted retries and transcription fallback to produce the required JSON.

## Run

Use a separate Python environment. Run from this country directory.

```bash
python -m pip install -r requirements.txt
mkdir -p outputs
python main.py
```

## Input and output

The official input contract and models/Qwen3-1.7B-Q8_0.gguf supplied through the official setup.

Obtain task materials from the organizer task card; preserve their original filenames. Generated results belong in `outputs/` or another directory outside Git.

## Files

- [`main.py`](main.py)
- [`requirements.txt`](requirements.txt)
- [`test-input.txt`](test-input.txt)
- [`test-output.txt`](test-output.txt)

Keep submission values on the organizer platform. No final submission value is supplied in these files.

## Model setup

The runtime expects `models/Qwen3-1.7B-Q8_0.gguf`. Obtain it through the official evaluator
or the [upstream GGUF repository](https://huggingface.co/Qwen/Qwen3-1.7B-GGUF):

```bash
huggingface-cli download Qwen/Qwen3-1.7B-GGUF Qwen3-1.7B-Q8_0.gguf --local-dir models
```

The audited model SHA-256 is `061b54daade076b5d3362dac252678d17da8c68f07560be70818cace6590cb1a`.
Install the specified llama-cpp-python build for the available device. The model is not
bundled in Git; it is about 1.8 GB. The small paired input/output files are organizer examples.
