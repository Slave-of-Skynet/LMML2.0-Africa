"""Run the unchanged official generation path on independently generated holdout questions."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--official", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    sys.path.insert(0, str(args.official.resolve()))
    from evaluation_script import check_adapter_file, check_reply, deregister, generate, load_llm, load_vocab, score_call

    v = load_vocab()
    check_adapter_file(args.model)
    llm = load_llm(str(args.model))
    records = json.loads(args.data.read_text(encoding="utf8"))[:args.limit]
    totals = Counter()
    for row in records:
        rep = generate(llm, row["prompt"])
        score = check_reply(rep, v)
        row["generated"] = rep
        row["plain"] = deregister(rep, v)
        row["expected_plain"] = deregister(row["reply"], v)
        row["style"] = score["conformance"]
        row["violations"] = score["rule_counts"]
        totals["examples"] += 1
        totals["fully_conforming"] += score["conformance"] == 1.0
        if row["type"] == "call":
            expected_seal = row["reply"].split()[-1].rstrip(".?")
            row["call_style"] = score_call(rep, expected_seal, v)[0] and len(rep.split()) == 3
            row["call_answer"] = row["plain"].rstrip(".?").casefold() == row["expected_plain"].rstrip(".?").casefold()
            totals["calls"] += 1
            totals["correct_calls"] += row["call_style"] and row["call_answer"]
    report = {"summary": dict(totals), "examples": records}
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps(totals, sort_keys=True))


if __name__ == "__main__":
    main()
