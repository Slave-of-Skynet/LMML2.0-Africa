# Ethiopia — The Azmari's Borrowed Melody

Estimate spectral recording profiles and fit splice boundaries to locate the borrowed interval.

## Run

Use a separate Python environment. Run from this country directory.

```bash
python -m pip install -r requirements.txt
mkdir -p outputs
python main.py /path/to/recording.wav
```

## Input and output

The organizer WAV recordings, CSV manifest and evaluator.

Obtain task materials from the organizer task card; preserve their original filenames. Generated results belong in `outputs/` or another directory outside Git.

Organizer material: [source 1](https://drive.google.com/drive/folders/1AH_ZNPVpkA5hDu9uvMm32Gyu__iYg9w7).

## Files

- [`main.py`](main.py)
- [`requirements.txt`](requirements.txt)

Keep submission values on the organizer platform. No final submission value is supplied in these files.
