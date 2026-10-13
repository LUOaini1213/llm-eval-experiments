"""Freeze the E4 (Starter Kit) test set: data/sk/items.jsonl, from the seed files in data/sk/ and E1's building-code gold.

No model or application is needed. CI rebuilds the file and fails if a byte changes.
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evalab import starterkit as SKL  # noqa: E402


def main():
    fm_gold = [json.loads(line) for line in (ROOT / "data" / "e1" / "fm_gold.jsonl").read_text(encoding="utf-8").splitlines()]
    items = SKL.build_items(fm_gold, SKL.load_seeds())
    out = ROOT / "data" / "sk" / "items.jsonl"
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print("wrote", out, len(items))
    for (risk, test), n in sorted(Counter((it["risk"], it["test"]) for it in items).items()):
        print(f"  {risk:22s} {test:32s} {n}")


if __name__ == "__main__":
    main()
