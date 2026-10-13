"""E4 component test (Starter Kit 3.6.3): sg-transit-geo-assistant's question screen, tested in isolation.

The screen (backend/app/guard.py) is a list of regular expressions that runs before any model call. This script
renders the 14 attack techniques of data/sk/attacks.json with the four goals in data/sk/geo_attacks.json, and runs
the screen on them and on that repository's own benign questions. No model is involved, so the result is exact.

    GEO_ROOT=../sg-transit-geo-assistant python scripts/sk_geo_screen.py      # CI checks out the pinned commit

Output: results/sk/geo_screen.json.
"""
import importlib.util
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.dont_write_bytecode = True
from evalab import starterkit as SKL  # noqa: E402
from evalab.stable import stable_round  # noqa: E402
from evalab.stats import wilson  # noqa: E402

GEO = Path(os.environ.get("GEO_ROOT", ROOT.parent / "sg-transit-geo-assistant")).resolve()


def load_screen(geo: Path):
    spec = importlib.util.spec_from_file_location("geo_guard", geo / "backend" / "app" / "guard.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.screen


def benign_questions(geo: Path) -> list[dict]:
    out = []
    for name, keep in (("questions.jsonl", "answerable"), ("heldout_2026-09-28.jsonl", "benign")):
        for line in (geo / "eval" / name).read_text(encoding="utf-8").splitlines():
            if line.strip():
                q = json.loads(line)
                if q.get("kind") == keep:
                    out.append({"id": q["id"], "source": name, "prompt": q["question"]})
    return out


def run(screen, attacks: dict, geo_goals: dict, benign: list[dict]) -> dict:
    rows = []
    for tk, tech in attacks["techniques"].items():
        for gk, goal in geo_goals["goals"].items():
            prompt = SKL.render_attack(tech, goal, attacks["leet"])
            rows.append({"id": f"geo-{tk}-{gk}", "technique": tk, "family": tech["family"], "goal": gk,
                         "prompt": prompt, "blocked_by": screen(prompt)})
    ben = [{**b, "blocked_by": screen(b["prompt"])} for b in benign]
    by_tech = defaultdict(lambda: [0, 0])
    by_goal = defaultdict(lambda: [0, 0])
    for r in rows:
        by_tech[r["technique"]][0] += r["blocked_by"] is not None
        by_tech[r["technique"]][1] += 1
        by_goal[r["goal"]][0] += r["blocked_by"] is not None
        by_goal[r["goal"]][1] += 1
    tp = sum(r["blocked_by"] is not None for r in rows)
    fp = sum(b["blocked_by"] is not None for b in ben)
    summary = {
        "attacks": len(rows), "attacks_blocked": tp, "attack_block_rate": tp / len(rows),
        "attack_block_rate_ci95": wilson(tp, len(rows)),
        "benign": len(ben), "benign_blocked": fp, "false_positive_rate": fp / len(ben),
        "false_positive_rate_ci95": wilson(fp, len(ben)),
        "by_technique": {k: {"name": attacks["techniques"][k]["name"], "blocked": v[0], "n": v[1]}
                         for k, v in by_tech.items()},
        "by_goal": {k: {"name": geo_goals["goals"][k]["name"], "blocked": v[0], "n": v[1]} for k, v in by_goal.items()},
    }
    return {"summary": summary, "attacks": rows, "benign": ben}


def main():
    head = subprocess.run(["git", "-C", str(GEO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    seeds = SKL.load_seeds()
    geo_goals = json.loads((ROOT / "data" / "sk" / "geo_attacks.json").read_text(encoding="utf-8"))
    if head != geo_goals["app_commit"]:
        raise SystemExit(f"sg-transit-geo-assistant is at {head!r}, expected {geo_goals['app_commit']}")
    res = run(load_screen(GEO), seeds["attacks"], geo_goals, benign_questions(GEO))
    res = {"app": "sg-transit-geo-assistant", "app_commit": head, **res}
    out = ROOT / "results" / "sk" / "geo_screen.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stable_round(res), indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    s = res["summary"]
    print(f"attacks blocked {s['attacks_blocked']}/{s['attacks']}, benign blocked {s['benign_blocked']}/{s['benign']}")
    for k, v in s["by_technique"].items():
        print(f"  {k} {v['name']:40s} {v['blocked']}/{v['n']}")


if __name__ == "__main__":
    main()
