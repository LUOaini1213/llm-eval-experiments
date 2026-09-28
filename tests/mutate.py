"""Mutation check: each deliberate bug below must make at least one unit test fail."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = str(ROOT / "tests")
S = "src/evalab/"
MUTANTS = [
    # statistics
    (S + "stats.py", "pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)",
     "pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n + 1)"),
    (S + "stats.py", "p_i = ((m * m).sum(axis=1) - r) / (r * (r - 1))", "p_i = ((m * m).sum(axis=1) - r) / (r * r)"),
    (S + "stats.py", "p = 2 * sum(binom_pmf(i, n) for i in range(k + 1))", "p = 2 * sum(binom_pmf(i, n) for i in range(k))"),
    (S + "stats.py", "running = max(running, (m - rank) * pvalues[i])", "running = (m - rank) * pvalues[i]"),
    (S + "stats.py", "centre = (p + z * z / (2 * n)) / den", "centre = p"),
    (S + "stats.py", "    d = x - y\n", "    d = x + y\n"),
    (S + "stats.py", "zb * math.sqrt(p_disc - diff ** 2)", "zb * math.sqrt(p_disc)"),
    # judge prompts, parsing and aggregation
    (S + "judge.py", 'return m.group(1).upper() == "CORRECT"', 'return m.group(1).upper() != "INCORRECT"'),
    (S + "judge.py", "return sum(1 for v in votes if v is True) * 2 > len(votes)",
     "return sum(1 for v in votes if v is True) * 2 >= len(votes)"),
    (S + "judge.py", "all(v is True for v in votes)", "any(v is True for v in votes)"),
    (S + "judge.py", "vals = [s if s is not None else 1 for s in scores]", "vals = [s for s in scores if s is not None]"),
    (S + "judge.py", 'if order == "AB" else (pair["second"], pair["first"])',
     'if order == "BA" else (pair["second"], pair["first"])'),
    (S + "judge.py", '_item_block(item, condition != "bin_free")', "_item_block(item, True)"),
    (S + "jury.py", "agg = AGGREGATIONS[aggregation]", "agg = J.majority"),
    # deterministic gold
    (S + "gold.py", 'return "ambiguous" if others else "correct"', 'return "correct"'),
    (S + "gold.py", "for k in (1000, 100, 0.01, 0.001)]", "for k in (10, 1000, 100, 0.01, 0.001)]"),
    (S + "gold.py", "pool = [float(x) for x in PCT.findall(answer)] if m.group(2) else nums", "pool = nums"),
    (S + "gold.py", 'return "ambiguous" if found - {gold} else "correct"', 'return "correct"'),
    (S + "gold.py", "return \" \".join(ROAD_ABBR.get(w, w) for w in norm(text).split())",
     "return \" \".join(w for w in norm(text).split())"),
    (S + "items.py", "excl |= {n for n in G.numbers(a) if n >= 1000} if float(gold) < 1000 else set()", "pass"),
    # E3 generator and its verification
    (S + "sgfacts.py", 'float(s["boundary_dist_m"]) >= min_boundary_m', 'float(s["boundary_dist_m"]) >= 0'),
    (S + "sgfacts.py", "        if str(want) != it.gold:\n", "        if False:\n"),
    (S + "moonshot_io.py", "        gold = gold.upper()\n", "        gold = gold\n"),
    # dataset audit
    (S + "audit.py", "elif text and not _same(opts[letter], text):", "elif False:"),
    (S + "audit.py", '"exact_duplicate" if keys[a][1] == keys[b][1] else "same_input_different_target"',
     '"exact_duplicate"'),
    (S + "audit.py", 'body = statement(ex["input"])', 'body = ex["input"]'),
    # pipeline analysis
    (S + "analysis.py", 'fa = sum(p.decisions[it["item_id"]] for it in neg)',
     'fa = sum(p.decisions[it["item_id"]] for it in pos)'),
    (S + "analysis.py", "if FAMILY[j] not in fams:", "if True:"),
    # amendment 1, analysis A: rankings and flips
    (S + "ranking.py", "flips = int(((sg * sj) < 0).sum())", "flips = int(((sg * sj) <= 0).sum())"),
    (S + "ranking.py", "den = np.sqrt((n0 - n1) * (n0 - n2).astype(float))",
     "den = np.sqrt((n0 - n1) * (n0 - n1).astype(float))"),
    (S + "ranking.py", "ties = int((untied & (sj == 0)).sum())", "ties = int((sj == 0).sum())"),
    (S + "ranking.py", "return np.bincount(system, weights=weight * passed, minlength=n_systems) / w_sum",
     "return np.bincount(system, weights=passed, minlength=n_systems) / w_sum"),
    (S + "ranking.py", "np.where(np.abs(d) < TOL, 0, np.sign(d))", "np.where(np.abs(d) < 0, 0, np.sign(d))"),
    (S + "ranking.py", "wp[rng.integers(0, len(wp), size=n_answers - k[i])]",
     "rp[rng.integers(0, len(rp), size=n_answers - k[i])]"),
    ("scripts/e1_ranking.py", '/ n_main[(it["source"], it["gold"])] for it in items])',
     '* 0 + 1.0 for it in items])'),
    # amendment 1, analysis B: weighted voting, Dawid-Skene, cross-validation
    (S + "aggregate.py", 'score = (params["weight"] * (2 * v - 1)).sum(axis=1)', 'score = (params["weight"] * v).sum(axis=1)'),
    (S + "aggregate.py", "w_fail = np.log1p(-se) - np.log(sp)", "w_fail = 0 * sp"),
    (S + "aggregate.py", "se = (v[pos].sum(axis=0) + 1) / (pos.sum() + 2)", "se = (v[pos].sum(axis=0) + 1) / (pos.sum() + 1)"),
    (S + "aggregate.py", "sp = (((1 - t)[:, None] * (1 - v)).sum(axis=0) + 1)", "sp = (((1 - t)[:, None] * v).sum(axis=0) + 1)"),
    (S + "aggregate.py", "t = np.where(maj > 0.5, 1.0, np.where(maj < 0.5, 0.0, 0.5))", "t = np.full(n, 0.5)"),
    (S + "aggregate.py", "                fold[i] = f\n", "                fold[i] = (f + i) % k\n"),
    (S + "aggregate.py", "tr = np.flatnonzero(folds != f)", "tr = np.arange(len(y))"),
    (S + "aggregate.py", "{s: fit_two_coin(v[m == s], y[m == s])", "{s: fit_two_coin(v, y)"),
    ("scripts/e1_aggregation.py", 'BEST = "qwen3.5:4b"', 'BEST = "gemma3:4b"'),
    # Moonshot connector
    ("moonshot_ext/connectors/ollama-connector.py", '            self.base_url = self.base_url[:-3]\n',
     "            pass\n"),
    ("moonshot_ext/connectors/ollama-connector.py", '"think": bool(params.get("think", False))',
     '"think": bool(params.get("think", True))'),
]

# The unmutated code must pass, otherwise every mutant would look "killed".
base = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", TESTS], capture_output=True,
                      cwd=ROOT)
assert base.returncode == 0, "tests fail on the unmutated code; fix them before running the mutation check"

escaped = 0
for fname, old, new in MUTANTS:
    f = ROOT / fname
    raw = f.read_bytes()
    text = raw.decode("utf-8")
    assert text.count(old) == 1, f"mutation target not unique in {fname}: {old[:70]}"
    f.write_bytes(text.replace(old, new).encode("utf-8"))
    compiled = subprocess.run([sys.executable, "-m", "py_compile", str(f)], capture_output=True)
    assert compiled.returncode == 0, f"mutant does not compile, so it proves nothing: {old[:70]}"
    try:
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", TESTS],
                           capture_output=True, text=True, cwd=ROOT)
    finally:
        f.write_bytes(raw)
    killed = r.returncode != 0
    escaped += not killed
    print(f"{'killed ' if killed else 'ESCAPED'}  {fname}: {old[:70]}", flush=True)
print(f"{len(MUTANTS) - escaped}/{len(MUTANTS)} mutants killed")
sys.exit(1 if escaped else 0)
