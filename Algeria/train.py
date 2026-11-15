"""Train assistant-only PEFT weights using the official prompt and local SmolLM2 files."""
import argparse
import json
import math
import random
import sys
from pathlib import Path

BASE = "HuggingFaceTB/SmolLM2-360M-Instruct"
# The shipped converter does not permute q/k projection output rows for GGUF.
TARGETS = ["v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def encode(tokenizer, build_prompt, row, max_length):
    prompt = build_prompt(row["prompt"])
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    ids = tokenizer.encode(prompt + row["reply"] + "<|im_end|>", add_special_tokens=False)
    if ids[:len(prefix)] != prefix or ids[-1] != tokenizer.eos_token_id:
        raise ValueError("Unexpected assistant token boundary or EOS")
    if len(ids) > max_length:
        raise ValueError(f"Sequence ({len(ids)}) exceeds max length {max_length}")
    return ids, len(prefix)


def collate(batch, torch, pad, device):
    size = max(len(ids) for ids, _ in batch)
    ids = torch.full((len(batch), size), pad, dtype=torch.long)
    mask = torch.zeros_like(ids)
    targets = torch.full_like(ids, -100)
    for i, (seq, prefix) in enumerate(batch):
        ids[i, :len(seq)] = torch.tensor(seq)
        mask[i, :len(seq)] = 1
        targets[i, prefix:len(seq)] = ids[i, prefix:len(seq)]
    return tuple(t.to(device) for t in (ids, mask, targets))


def assistant_loss(model, batch, torch):
    input_ids, mask, targets = batch
    core = model.get_base_model()
    hidden = core.model(input_ids=input_ids, attention_mask=mask, use_cache=False).last_hidden_state
    supervised = targets[:, 1:] != -100
    logits = core.lm_head(hidden[:, :-1][supervised]).float()
    return torch.nn.functional.cross_entropy(logits, targets[:, 1:][supervised])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--official", type=Path, required=True)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resume-adapter", type=Path)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=144)
    ap.add_argument("--rank", type=int, default=12)
    ap.add_argument("--alpha", type=int, default=24)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--checkpoint-every", type=int, default=0)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = ap.parse_args()
    if min(args.steps, args.batch, args.rank) < 1:
        ap.error("steps, batch and rank must be positive")
    sys.path.insert(0, str(args.official.resolve()))
    from evaluation_script import build_prompt

    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    rng = random.Random(args.seed)
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    print(f"torch={torch.__version__} device={device} dtype={dtype}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=dtype, device_map={"": device}, use_safetensors=True, trust_remote_code=False,
    )
    if args.resume_adapter:
        model = PeftModel.from_pretrained(model, args.resume_adapter, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(
            r=args.rank, lora_alpha=args.alpha, lora_dropout=0.0, task_type="CAUSAL_LM",
            target_modules=TARGETS,
        ))
    model.print_trainable_parameters()
    model.train()
    rows = json.loads(args.data.read_text(encoding="utf8"))
    pairs = [encode(tokenizer, build_prompt, row, args.max_length) for row in rows]
    if not pairs:
        raise ValueError("Training data is empty")
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, fused=device == "cuda")
    print(f"examples={len(pairs)} max_length={max(len(ids) for ids, _ in pairs)}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)

    cursor = len(pairs)
    exposure_count = 0
    losses = []
    for step in range(1, args.steps + 1):
        if cursor >= len(pairs):
            rng.shuffle(pairs)
            cursor = 0
        examples = pairs[cursor:cursor + args.batch]
        cursor += len(examples)
        exposure_count += len(examples)
        batch = collate(examples, torch, tokenizer.pad_token_id, device)
        with torch.autocast(device_type=device, dtype=dtype, enabled=device == "cuda" and dtype == torch.bfloat16):
            loss = assistant_loss(model, batch, torch)
        if not math.isfinite(value := loss.item()):
            raise RuntimeError(f"Non-finite loss at step {step}: {value}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
        opt.step()
        losses.append(value)
        if step == 1 or step % 10 == 0 or step == args.steps:
            print(f"step={step} loss={value:.4f}", flush=True)
        if args.checkpoint_every and step % args.checkpoint_every == 0 and step < args.steps:
            model.save_pretrained(args.out / f"step-{step}")
    model.save_pretrained(args.out)
    config = model.peft_config["default"]
    metrics = {
        "base": args.base,
        "examples": len(pairs),
        "steps": args.steps,
        "batch_size": args.batch,
        "sample_exposures": exposure_count,
        "effective_epochs": exposure_count / len(pairs),
        "rank": config.r,
        "alpha": config.lora_alpha,
        "target_modules": sorted(config.target_modules),
        "learning_rate": args.lr,
        "seed": args.seed,
        "device": device,
        "dtype": str(dtype),
        "torch": torch.__version__,
        "first_step_loss": losses[0],
        "last_step_loss": losses[-1],
    }
    (args.out / "run.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf8")
    print(f"Saved PEFT weights to {args.out}", flush=True)


if __name__ == "__main__":
    main()
