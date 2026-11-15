# Tanzania — Please, You Are So Delusional!

Retrieve whole notices using explicit title context and a pinned shared sentence encoder.

## Run

Use a separate Python environment. Run from this country directory.

```bash
python -m pip install -r requirements.txt
mkdir -p outputs
python retrieval_pipeline.py --corpus /path/to/corpus.jsonl --questions /path/to/questions.json --output outputs/retrieved_ids.json
```

## Input and output

corpus.jsonl and questions.json; the pinned sentence encoder downloads on first use unless cached.

Obtain task materials from the organizer task card; preserve their original filenames. Generated results belong in `outputs/` or another directory outside Git.

Organizer material: [source 1](https://drive.google.com/drive/folders/1E94rYgT1d4vum8zPRHr-TMATz4NxJssb).

## Files

- [`flag_utils.py`](flag_utils.py)
- [`requirements.txt`](requirements.txt)
- [`retrieval_pipeline.py`](retrieval_pipeline.py)
- [`retrieved_ids.json`](retrieved_ids.json)

Keep submission values on the organizer platform. No final submission value is supplied in these files.
