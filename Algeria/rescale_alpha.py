"""Copy a PEFT LoRA checkpoint with a different lora_alpha (scales the update by alpha/r)."""
import argparse
import json
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--alpha", type=float, required=True)
    args = ap.parse_args()
    args.dst.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.src / "adapter_model.safetensors", args.dst / "adapter_model.safetensors")
    config = json.loads((args.src / "adapter_config.json").read_text(encoding="utf8"))
    config["lora_alpha"] = args.alpha
    (args.dst / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf8")
    print(f"r={config['r']} alpha={args.alpha} -> {args.dst}")


if __name__ == "__main__":
    main()
