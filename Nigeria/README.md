# Nigeria — The Gángan Never Lies

Compute a Hann-window spectrogram and render the 2500-4000 Hz message band.

## Run

Use a separate Python environment. Run from this country directory.

```bash
python -m pip install -r requirements.txt
mkdir -p outputs
python main.py /path/to/challenge.wav --output-dir outputs
```

## Input and output

challenge.wav, mono uncompressed 16-bit PCM; the script validates the audio contract.

Obtain task materials from the organizer task card; preserve their original filenames. Generated results belong in `outputs/` or another directory outside Git.

## Files

- [`main.py`](main.py)
- [`requirements.txt`](requirements.txt)

Keep submission values on the organizer platform. No final submission value is supplied in these files.
