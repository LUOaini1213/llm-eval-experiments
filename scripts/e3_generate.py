"""Generate the Singapore bus-network test set, verify every gold answer, and write it out.

Outputs:
- data/e3/sgbus_items.jsonl       the items (question, gold, accepted forms, template, difficulty, stratum)
- results/e3/generation.json      counts by template, difficulty and stratum, and the verification result
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evalab.items import write_jsonl  # noqa: E402
from evalab.sgfacts import generate, to_dicts, verify_items  # noqa: E402

N, SEED = 180, 20260928


def main():
    items = generate(N, seed=SEED)
    problems = verify_items(items)
    if problems:
        raise SystemExit("verification failed:\n" + "\n".join(problems))
    write_jsonl(ROOT / "data" / "e3" / "sgbus_items.jsonl", to_dicts(items))
    summary = {"n": len(items), "seed": SEED, "verified": len(items), "verification_problems": problems,
               "by_template": Counter(i.template for i in items), "by_difficulty": Counter(i.difficulty for i in items),
               "by_stratum": Counter(i.stratum for i in items),
               "distinct_gold_share_max": {t: max(Counter(i.gold for i in items if i.template == t).values()) /
                                           sum(1 for i in items if i.template == t)
                                           for t in sorted({i.template for i in items})}}
    out = ROOT / "results" / "e3" / "generation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1), encoding="utf-8", newline="\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
