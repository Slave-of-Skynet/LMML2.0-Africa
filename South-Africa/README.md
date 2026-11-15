# South Africa — Ngiyakuzwa?

Denoise the stadium recording and run local speech transcription with a pinned checkpoint.

## Run

Use a separate Python environment. Run from this country directory.

```bash
python -m pip install -r requirements.txt
mkdir -p outputs
python main.py /path/to/audio.wav --out outputs
```

## Input and output

The organizer audio file; the pinned speech checkpoint downloads on first use unless cached.

Obtain task materials from the organizer task card; preserve their original filenames. Generated results belong in `outputs/` or another directory outside Git.

Organizer material: [source 1](https://drive.google.com/file/d/1U-gw6IifI8Js7rDT3wJNsmhDzOYvYAae/view).

## Files

- [`main.py`](main.py)
- [`requirements.txt`](requirements.txt)

Keep submission values on the organizer platform. No final submission value is supplied in these files.
