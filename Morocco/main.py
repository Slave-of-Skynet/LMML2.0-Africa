"""Morocco (The Dispatch Desk of Fez): route each request to the cheap or the expensive guide.

Idea: the requests come in a few fixed kinds. On the 400 training requests we measured how often
each guide is right for each kind, and send a request to the cheap guide only where that costs
(almost) no correct answers:

  1. plain price lookup            cheap 45/45 right            -> cheap, nothing lost
  2. shortest walk where a detour  cheap 1/30, expensive 3/30   -> cheap, both fail anyway
     beats the direct street
  3. "which souk keeps the least"  cheap 4/15, expensive 6/15   -> cheap, little lost
  4. "carry in total", <=12 crates cheap 20/22 right            -> cheap, little lost
  everything else (and anything unrecognised)                   -> expensive

Training result: quality 0.84 (need >= 0.80), cost 18.3 dirham per request (need <= 20).
Standard library only; the guides themselves are never called here.
"""
import csv
import re


def detour_is_shorter(prompt):
    """True when some two-street route is shorter than the direct street (or there is no direct one)."""
    edges = {}
    for a, b, minutes in re.findall(r"([A-Z][a-z]+)-([A-Z][a-z]+) (\d+) minutes", prompt):
        edges[(a, b)] = edges[(b, a)] = int(minutes)
    ends = re.search(r"from ([A-Z][a-z]+) to ([A-Z][a-z]+)", prompt)
    if not ends:
        return False
    start, goal = ends.groups()
    direct = edges.get((start, goal))
    places = {p for pair in edges for p in pair}
    via = [edges[(start, m)] + edges[(m, goal)] for m in places if (start, m) in edges and (m, goal) in edges]
    return bool(via) and (direct is None or min(via) < direct)


def route(prompt):
    if re.search(r"What (do|does) [a-z ]+ cost\?", prompt):
        return "cheap"
    if "shortest walk" in prompt and detour_is_shorter(prompt):
        return "cheap"
    if "keeps the least" in prompt:
        return "cheap"
    crates = re.search(r"loads (\d+) crates", prompt)
    if "carry in total" in prompt and crates and int(crates.group(1)) <= 12:
        return "cheap"
    return "expensive"


def main():
    with open("test-input.csv", newline="", encoding="utf-8") as src, \
         open("routing_log.csv", "w", newline="", encoding="utf-8") as dst:
        out = csv.writer(dst)
        out.writerow(["request_id", "model"])
        for row in csv.DictReader(src):
            out.writerow([row["request_id"], route(row["prompt"])])


if __name__ == "__main__":
    main()
