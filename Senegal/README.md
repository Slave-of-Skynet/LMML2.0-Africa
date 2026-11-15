# Senegal — The Teranga FM Token Diet

Build a low-cost routing configuration and answer through the official fact-bank contract.

## Run

Use a separate Python environment. Run from this country directory.

```bash
python -m pip install -r requirements.txt
mkdir -p outputs
python make_config.py
python main.py --config routing_config.json --output answers.json
```

## Input and output

The official dataset directory containing the fact bank and request data.

Obtain task materials from the organizer task card; preserve their original filenames. Generated results belong in `outputs/` or another directory outside Git.

Organizer material: [source 1](https://drive.google.com/drive/folders/1QuDKEu05swvKZiz_SDCzyLaEyKNT8iud?usp=sharing).

## Files

- [`main.py`](main.py)
- [`make_config.py`](make_config.py)
- [`requirements.txt`](requirements.txt)
- [`routing_config.json`](routing_config.json)

Keep submission values on the organizer platform. No final submission value is supplied in these files.

Required shared helper: [`model_utils.py`](model_utils.py). The bundled [`dataset/fact_bank.json`](dataset/fact_bank.json) supplies the official fact-bank contract.
