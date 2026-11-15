"""Generate independent register instruction examples with labels checked by the organizers' code."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import re
import sys

NAME = re.compile(r"\{([^{}|]+)\|([A-F])\}")
TOKEN = re.compile(r"^([^\w]*)([\w'-]+)([^\w]*)$")
SENTENCE = re.compile(r"(?<=[.?])\s+")
ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
TENS = "twenty thirty forty fifty sixty seventy eighty ninety".split()
COUNT_WORDS = dict(zip(ONES, range(20))) | dict(zip(TENS, range(20, 100, 10)))
CHAINS = {"hundred": 100, "thousand": 1000, "million": 1000000, "billion": 1000000000}
ONES_PATTERN = "(?:" + "|".join(ONES[1:10]) + ")"
SMALL_PATTERN = "(?:" + "|".join(TENS) + ")(?:[- ]" + ONES_PATTERN + ")?|(?:" + "|".join(ONES) + ")"
HUNDRED_PATTERN = r"(?:a|" + ONES_PATTERN + r") hundred(?: (?:and )?(?:" + SMALL_PATTERN + "))?"
GROUP_PATTERN = "(?:" + HUNDRED_PATTERN + "|" + SMALL_PATTERN + "|a)"
COUNT_PATTERN = re.compile(
    r"\{[^{}]+\}|\b(?:" + GROUP_PATTERN + r" (?:billion|million|thousand)"
    r"(?: (?:and )?" + GROUP_PATTERN + r" (?:million|thousand))*"
    r"(?: (?:and )?(?:" + HUNDRED_PATTERN + "|" + SMALL_PATTERN + "))?"
    "|" + HUNDRED_PATTERN + "|" + SMALL_PATTERN + r")\b", re.IGNORECASE,
)
IRREGULAR = {
    "men": "man", "women": "woman", "children": "child", "people": "people",
    "persons": "person", "oases": "oasis", "knives": "knife",
}
CALL_SUFFIXES = (
    " Answer with one noun.",
    " Give just the noun.",
    " Reply using only a single noun.",
    " Give a one-noun answer.",
)
HOLDOUT_SUFFIX = " Respond with a single noun only."
DEFINITIONS = {
    "A": [
        ("sea", "a large body of salty water"),
        ("stone", "a hard piece of rock"),
        ("river", "a natural flow of water leading to a lake or ocean"),
        ("forest", "an area covered by many trees"),
        ("water", "the liquid that freezes into ice"),
        ("beach", "the sandy strip next to the sea"),
        ("mountain", "a natural elevation with a high summit"),
        ("island", "land surrounded by water"),
        ("cave", "a natural underground cavity"),
        ("gold", "a precious yellow metal"),
        ("lake", "a body of water surrounded by land"),
        ("desert", "a very dry place with little rainfall"),
        ("cloud", "a mass of droplets suspended in the sky"),
        ("valley", "a low area between hills"),
        ("wind", "moving air"),
    ],
    "B": [
        ("cat", "a domestic animal that meows"),
        ("dog", "a domestic animal that barks"),
        ("fox", "a wild animal with a bushy tail and a pointed muzzle"),
        ("teacher", "a person who helps students learn"),
        ("doctor", "a person trained to treat illness"),
        ("sheep", "a farm animal known for its wool"),
        ("lion", "a large cat whose male often has a mane"),
        ("king", "a male monarch"),
        ("queen", "a female monarch"),
        ("soldier", "a person serving in an army"),
        ("farmer", "a person who grows crops or raises livestock"),
        ("poet", "a person who writes verse"),
        ("camel", "a desert animal with humps"),
        ("goat", "a sure-footed farm animal often kept for its milk"),
        ("scientist", "a person who tests ideas through systematic observation"),
    ],
    "C": [
        ("bridge", "a structure for crossing over a river"),
        ("boat", "a small vehicle that travels on water"),
        ("car", "a road vehicle people drive to work or home"),
        ("train", "a vehicle with connected cars running on rails"),
        ("school", "a place where students attend lessons"),
        ("airport", "a place where passenger planes take off"),
        ("museum", "a place that preserves and exhibits collections"),
        ("hospital", "a building where patients receive care"),
        ("door", "a movable barrier at the entrance to a room"),
        ("castle", "a fortified residence with defensive walls"),
        ("road", "a route built for vehicles"),
        ("knife", "a handheld tool with a sharp blade"),
        ("tower", "a tall building or part of a building"),
        ("dam", "a structure that holds back a river"),
        ("village", "a rural settlement smaller than a town"),
    ],
    "D": [
        ("century", "a period of a hundred years"),
        ("decade", "a period of ten years"),
        ("week", "a period of seven days"),
        ("minute", "a period of sixty seconds"),
        ("hour", "a period of sixty minutes"),
        ("festival", "a public celebration often held each year"),
        ("war", "a prolonged armed conflict between states"),
        ("birthday", "the annual anniversary of a person's birth"),
        ("summer", "the warmest of the four seasons in temperate climates"),
        ("winter", "the coldest of the four seasons in temperate climates"),
        ("evening", "the time of day between late afternoon and night"),
        ("season", "a recurring division of the year with characteristic weather"),
    ],
    "E": [
        ("book", "a bound collection of pages"),
        ("alphabet", "an ordered set of letters used to write a language"),
        ("poem", "a piece of writing composed in verse"),
        ("film", "a work made of moving images shown on a screen"),
        ("song", "words set to music for a voice"),
        ("law", "a rule enforced by a government"),
        ("anthem", "a song that represents a country"),
        ("chemistry", "the science that studies how substances react"),
        ("physics", "the science that studies matter and energy"),
        ("novel", "a long fictional book"),
        ("picture", "an image such as a drawing or photograph"),
        ("language", "a structured system of communication such as {English|E}"),
        ("question", "a sentence used to ask for information"),
        ("word", "a unit of language such as a noun or verb"),
    ],
    "F": [
        ("bread", "a baked food made from dough"),
        ("cheese", "a solid food made from curdled milk"),
        ("coffee", "a drink made by brewing ground roasted beans"),
        ("tea", "a drink made by steeping dried leaves"),
        ("rice", "a staple grain often served boiled"),
        ("soup", "a liquid dish made by cooking ingredients in broth"),
        ("honey", "a sweet substance made by bees"),
        ("sugar", "sweet crystals often obtained from cane or beet"),
        ("salt", "a common white crystalline seasoning"),
        ("{silk|F}", "a fine fabric spun by silkworms"),
        ("coin", "a small metal piece used as money"),
        ("milk", "a white liquid produced by mammals to feed their young"),
        ("olive", "a small fruit pressed for oil in {Mediterranean|A} cooking"),
        ("egg", "a shelled food laid by a hen"),
    ],
}


def inflections(v):
    forms = dict.fromkeys(v["words"])
    for noun in v["words"]:
        for plural in (noun + "s", noun + "es", noun[:-1] + "ies" if noun.endswith("y") else ""):
            if plural in v["_forbidden"] and plural not in forms:
                forms[plural] = noun
    forms.update(IRREGULAR)
    return {word: v["words"][singular or word] for word, singular in forms.items()
            if (singular or word) in v["words"]}


def numeral(number, v):
    return "-".join(v["digits"][int(digit)] for digit in str(number)) + "-" + v["classes"]["D"]["suffix"]


def cardinal(match):
    value = group = 0
    for word in re.split(r"[- ]", match[0].lower()):
        if word == "hundred":
            group *= 100
        elif word in CHAINS:
            value += group * CHAINS[word]
            group = 0
        elif word != "and":
            group += 1 if word == "a" else COUNT_WORDS[word]
    return str(value + group)


def register(text, v, nouns, single=False):
    text = COUNT_PATTERN.sub(lambda m: m[0] if m[0].startswith("{") else cardinal(m), text)
    text = NAME.sub(lambda m: " ".join(
        word + "-" + v["classes"][m[2]]["suffix"] for word in m[1].split()), text)
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    sentences = []
    for sentence in SENTENCE.split(text.strip()):
        punctuation = sentence[-1] if sentence.endswith((".", "?")) else "."
        converted = []
        last_class = None
        for raw in sentence.rstrip(".?!").split():
            match = TOKEN.fullmatch(raw)
            if match is None:
                raise ValueError(f"Unsupported token {raw!r} in {text!r}")
            lead, word, trail = match.groups()
            possessive = ""
            if word.endswith("'s"):
                word, possessive = word[:-2], "'s"
            lower = word.lower()
            if word.isascii() and word.isdecimal():
                word, cls = numeral(word, v), "D"
            elif lower in nouns:
                info = nouns[lower]
                cls = info["class"]
                word = f"{info['stem']}-{v['classes'][cls]['suffix']}"
            else:
                cls = v["_suffix_to_class"].get(lower.rsplit("-", 1)[-1]) if "-" in lower else None
            if cls:
                last_class = cls
            converted.append(lead + word + possessive + trail)
        if single and (len(converted) != 1 or len(sentences) or last_class is None):
            raise ValueError(f"Call must contain exactly one class-marked noun: {converted}")
        seal = v["_class_to_seal"].get(last_class, v["neutral_seal"])
        sentences.append(" ".join(converted + [seal + punctuation]))
    return v["opening"] + ", " + " ".join(sentences)


def unannotate(value):
    return NAME.sub(r"\1", value)


def question_key(prompt):
    return " ".join(re.findall(r"\w+", prompt.casefold()))


def build_rows(v, check_reply, public, seed, expanded=False):
    rng = random.Random(seed)
    nouns = inflections(v)
    forbidden = {question_key(r["prompt"]) for r in public}
    seen = set()
    rows = {"train": [], "holdout": []}

    def add(split, kind, question, plain, single=False):
        key = question_key(question)
        if key in forbidden or key in seen:
            return
        answer = register(plain, v, nouns, single)
        checked = check_reply(answer, v)
        if checked["conformance"] != 1.0:
            raise ValueError((question, answer, checked["sentences"]))
        seen.add(key)
        rows[split].append({"prompt": question, "reply": answer, "type": kind})

    def call(split, prompt, noun, variants=2):
        endings = CALL_SUFFIXES if split == "train" else (HOLDOUT_SUFFIX,)
        for ending in (rng.sample(endings, variants) if split == "train" else endings):
            add(split, "call", prompt + ending, noun, True)

    path = Path(__file__).with_name("facts.tsv")
    groups = {}
    for line in path.read_text(encoding="utf8").splitlines():
        if line and not line.startswith("#"):
            group, question, noun, full = line.split("\t")
            groups.setdefault(group, []).append((question, noun, full))
    for group, facts in groups.items():
        rng.shuffle(facts)
        for i, (question, noun, full) in enumerate(facts):
            split = "holdout" if i < max(1, len(facts) // 9) else "train"
            add(split, group, question, full)
            if noun:
                call(split, question, noun)
                if split == "train":
                    add(split, "short_fact", question.rstrip("?") + "? Give a brief answer.", f"It is {noun}.")

    for cls, definitions in DEFINITIONS.items():
        definitions = definitions.copy()
        rng.shuffle(definitions)
        for i, (noun, definition) in enumerate(definitions):
            article = "an" if unannotate(noun)[0].lower() in "aeiou" or noun == "hour" else "a"
            if i < 2:
                add("holdout", "definition", f"Describe what {article} {unannotate(noun)} is.", f"It is {definition}.")
                call("holdout", f"Which noun describes {definition}?", noun)
                continue
            add("train", "definition", f"What is {article} {unannotate(noun)}?", f"It is {definition}.")
            call("train", f"What do you call {definition}?", noun)
            add("train", "definition", f"Give an example of {definition}.", f"{article.capitalize()} {noun} is an example.")

    # Hold out distinct arithmetic inputs and distinct prompt templates.
    for i, value in enumerate(rng.sample(range(1000, 2099), 120)):
        if i < 12:
            call("holdout", f"A plaque says {value}. Which year does it display?", str(value))
        else:
            # The reference is given in the question so this teaches conversion, not arbitrary facts.
            add("train", "numeral", f"An inscription is dated {value}. What year is written there?",
                f"The inscription is dated {value}.")
            if i % 5 == 0:
                call("train", f"A record is dated {value}. Which year is on it?", str(value), variants=1)
    pairs = [(a, b) for a in range(1, 35) for b in range(a, 36)]
    for i, (a, b) in enumerate(rng.sample(pairs, 145)):
        if i < 15:
            add("holdout", "arithmetic", f"Find the total of {a} and {b}.", f"Adding {a} and {b} gives {a+b}.")
        else:
            q = f"How many objects are there if you have {a} and get {b} more?"
            add("train", "arithmetic", q, f"There are {a+b} in total.")
            if i % 10 == 0:
                call("train", f"Which number is the sum of {a} and {b}?", str(a + b), variants=1)
    if expanded:
        # Reduce repeated training calls while keeping every distinct question
        # and all independently constructed holdout rows.
        used_calls = set()
        selected = []
        for row in rows["train"]:
            if row["type"] == "call":
                key = next((row["prompt"][:-len(s)] for s in CALL_SUFFIXES if row["prompt"].endswith(s)),
                           row["prompt"])
                if key in used_calls:
                    continue
                used_calls.add(key)
            selected.append(row)
        rows["train"] = selected
        more = Path(__file__).with_name("more_facts.tsv")
        for line in more.read_text(encoding="utf8").splitlines():
            if line and not line.startswith("#"):
                group, question, noun, full = line.split("\t")
                add("train", "extra_" + group, question, full)
                if noun:
                    call("train", question, noun, variants=1)
        # Mix the additions after independently choosing the reference split.
        rng.shuffle(rows["train"])

    for examples in rows.values():
        rng.shuffle(examples)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--expanded", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.official.resolve()))
    from style_checker import check_reply, load_vocab

    public = []
    for name in ("prompts", "calls"):
        public.extend(json.loads((args.official / "public" / f"{name}.json").read_text(encoding="utf8")))
    rows = build_rows(load_vocab(), check_reply, public, args.seed, args.expanded)
    args.out.mkdir(parents=True, exist_ok=True)
    for split, examples in rows.items():
        path = args.out / f"{split}.json"
        path.write_text(json.dumps(examples, ensure_ascii=False, indent=1) + "\n", encoding="utf8")
        print(f"{split}: {len(examples)} {dict(Counter(r['type'] for r in examples))}")


if __name__ == "__main__":
    main()
