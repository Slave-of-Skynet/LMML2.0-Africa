# Algeria — Here, at your service!

Generate checker-validated instruction data and train an assistant-token-only LoRA adapter with explicit conversion to the official runtime format.

## Run

Use a separate Python environment. Run from this country directory.

```bash
python -m pip install -r requirements.txt
mkdir -p outputs
python make_data.py --official /path/to/official --out outputs/data
# See the training recipe below for the base model, training and conversion.
```

## Input and output

The official evaluator, style checker and conversion tools, plus the pinned base model described below.

Obtain task materials from the organizer task card; preserve their original filenames. Generated results belong in `outputs/` or another directory outside Git.

Organizer material: [source 1](https://drive.google.com/drive/folders/1Rgycm8QaofJFRxem5I-3g0JHA4nmpp73?usp=sharing).

## Files

- [`check_holdout.py`](check_holdout.py)
- [`facts.tsv`](facts.tsv)
- [`make_data.py`](make_data.py)
- [`more_facts.tsv`](more_facts.tsv)
- [`requirements.txt`](requirements.txt)
- [`rescale_alpha.py`](rescale_alpha.py)
- [`train.py`](train.py)

Keep submission values on the organizer platform. No final submission value is supplied in these files.

## Training and conversion recipe

Keep the official evaluator folder and the base model outside Git. Use Python 3.12 and a
device-appropriate PyTorch installation. The base is `HuggingFaceTB/SmolLM2-360M-Instruct`,
revision `a10cc1512eabd3dde888204e902eca88bddb4951`. Run the official evaluator's setup first.

```bash
python /path/to/official/evaluation_script.py --setup
python make_data.py --official /path/to/official --out outputs/data
python -c "from huggingface_hub import snapshot_download; snapshot_download('HuggingFaceTB/SmolLM2-360M-Instruct', revision='a10cc1512eabd3dde888204e902eca88bddb4951', allow_patterns=['*.json','model.safetensors'], local_dir='outputs/base')"
python train.py --official /path/to/official --base outputs/base --data outputs/data/train.json --out outputs/peft --batch 6 --steps 200 --rank 12 --alpha 24 --checkpoint-every 64
python rescale_alpha.py outputs/peft/step-128 outputs/peft-rescaled --alpha 22.5
python /path/to/official/tools/lora_to_gguf.py outputs/peft-rescaled --outfile outputs/adapter.gguf
python /path/to/official/evaluation_script.py --model outputs/adapter.gguf --data /path/to/official/public --out outputs/evaluation.json
```

No trained adapter was present in the preserved source package. This repository supplies
the training/data recipe; it does not assert byte-identical reconstruction of a prior upload.
