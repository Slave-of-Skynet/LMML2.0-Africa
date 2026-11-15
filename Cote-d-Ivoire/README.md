# Côte d’Ivoire — The Abidjan Transit Ledger

Use the embedded TF-IDF vocabulary and multinomial logistic-regression coefficients to route all old and new query templates.

## Run

Use a separate Python environment. Run from this country directory.

```bash
python -m pip install -r requirements.txt
mkdir -p outputs
python main.py --input test-input.json --output predictions.json
```

## Input and output

test-input.json, a list of records with id and text. Model coefficients are embedded in main.py.

Obtain task materials from the organizer task card; preserve their original filenames. Generated results belong in `outputs/` or another directory outside Git.

## Files

- [`main.py`](main.py)
- [`requirements.txt`](requirements.txt)

Keep submission values on the organizer platform. No final submission value is supplied in these files.
