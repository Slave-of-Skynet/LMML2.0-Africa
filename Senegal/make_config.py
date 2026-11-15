"""Senegal (Token Diet): pick the cheapest routing per question.

Questions are billed independently, so the cheapest valid total is the sum of the
cheapest valid choice per question. There are only 2*2*2*2 = 16 choices each, so try them all.
"""
import itertools
import model_utils as mu

bank = mu.load_fact_bank()
config, total = {}, 0
for qid, q in bank.items():
    best = None
    for tier, style, fewshot, context in itertools.product(mu.TIERS, mu.STYLES, (False, True), (False, True)):
        call = mu.build_call(q, tier, style, fewshot, context)
        if mu.question_passes(q["slots"], call["answer"]) and (best is None or call["tokens"] < best[0]):
            best = (call["tokens"], dict(tier=tier, style=style, fewshot=fewshot, context=context))
    assert best, f"{qid}: no choice passes"
    config[qid] = best[1]; total += best[0]
    print(qid, best[0], best[1])
mu.save_json(config, mu.HERE / "routing_config.json")
check = mu.run_config(config, bank)
print("total tokens:", check["total_tokens"], "| quality gate:", check["quality_gate"], "| budget gate:", check["budget_gate"])
assert check["quality_gate"] and check["budget_gate"] and check["total_tokens"] == total
