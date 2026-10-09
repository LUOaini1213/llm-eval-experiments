"""Mutation check: compileable defects must cause real pytest test failures.

Run without arguments for the full clean baseline and mutation suite, or use --self-test
to check the harness in temporary toy projects without touching repository sources.
Collection/setup errors, missing reports, crashes and timeouts are infrastructure errors,
not mutation kills. Each subprocess gets a fresh bytecode location and every source write
is covered by restoration, including compilation and test-launch failures.
"""
import os
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = str(ROOT / "tests")
TEST_TIMEOUT_S = 300
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
    # Review regressions: missing replies are not votes; subsets carry only their own costs.
    (S + "analysis.py",
     '    if key not in idx:\n        raise ValueError(f"Missing judgment: {key}")\n    return idx[key]',
     '    if key not in idx:\n        return dict(parsed=False, prompt_tokens=0, output_tokens=0, seconds=0)\n'
     '    return idx[key]'),
    (S + "analysis.py",
     '        if key in idx:\n            raise ValueError(f"Duplicate judgment: {key}")\n        _validate_record(r)',
     '        _validate_record(r)'),
    (S + "analysis.py", "    if missing:\n", "    if False:\n"),
    (S + "analysis.py",
     '    cost = {k: sum(p.item_cost[it["item_id"]][k] for it in items)\n'
     '            for k in ("calls", "prompt_tokens", "output_tokens", "seconds")}',
     '    cost = p.cost'),
    # Exclude identifier/date occurrences, while retaining independently stated clock values.
    (S + "gold.py", "match.span() in identifiers", "False"),
    (S + "gold.py", "any(start <= match.start() < end for start, end in date_spans)", "False"),
    # Appending an amendment preserves the original hash; changing the original does not.
    (S + "experiment.py", "return raw[:raw.index(marker) + 1] if marker in raw else raw", "return raw"),
    (S + "experiment.py",
     'if manifest.get("preregistration_sha256") != preregistration_sha256(prereg):', 'if False:'),
]

class HarnessError(RuntimeError):
    """The run supplies no valid evidence about a mutant."""


def replacement(root, mutant):
    fname, old, new = mutant
    path = root / fname
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    if text.count(old) != 1 or old == new:
        raise HarnessError(f"mutation target must occur once and change: {fname}: {old[:70]}")
    return path, raw, text.replace(old, new).encode("utf-8")


def compile_source(raw, path):
    # Compile without importing the module or creating a .pyc next to product sources.
    try:
        compile(raw, str(path), "exec")
    except (SyntaxError, ValueError) as exc:
        raise HarnessError(f"mutant does not compile; not a kill: {path}: {exc}") from exc


def run_tests(root, tests, output_dir, *, fail_fast=False, timeout=TEST_TIMEOUT_S):
    output_dir.mkdir(parents=True)
    report = output_dir / "pytest.xml"
    env = dict(os.environ)
    env.pop("PYTEST_ADDOPTS", None)  # always run the declared suite, not a shell's -k filter
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(root / "src"), str(root), env.get("PYTHONPATH")]))
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPYCACHEPREFIX"] = str(output_dir / "pycache")
    cmd = [sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
           f"--junitxml={report}", str(tests)]
    if fail_fast:
        cmd.append("-x")
    try:
        result = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise HarnessError(f"pytest timed out after {timeout}s; not a kill") from exc
    output = result.stdout + result.stderr
    (output_dir / "pytest.log").write_text(output, encoding="utf-8")
    try:
        document = ET.parse(report)
    except (OSError, ET.ParseError) as exc:
        raise HarnessError(f"pytest exited {result.returncode} without a valid report; not a kill:\n"
                           + output[-3000:]) from exc
    cases = document.findall(".//testcase")
    errors = document.findall(".//error")
    skipped = sum(case.find("skipped") is not None for case in cases)
    failures = []
    for case in cases:
        failure = case.find("failure")
        if failure is not None:
            name = f"{case.get('classname', '')}::{case.get('name', '')}"
            message = " ".join((failure.get("message") or failure.text or "test failure").split())
            failures.append(f"{name}: {message[:180]}")
    if errors or not cases or len(cases) == skipped or result.returncode not in (0, 1):
        raise HarnessError(f"pytest exited {result.returncode}, {len(errors)} errors, "
                           f"{len(cases)} cases/{skipped} skipped; not a kill:\n" + output[-3000:])
    if (result.returncode == 1) != bool(failures):
        raise HarnessError(f"pytest exit/report disagree; not a kill:\n" + output[-3000:])
    return {"killed": bool(failures), "tests": len(cases), "skipped": skipped, "failures": failures}


def run_mutant(root, tests, mutant, output_dir, *, timeout=TEST_TIMEOUT_S):
    path, raw, changed = replacement(root, mutant)
    try:
        path.write_bytes(changed)
        compile_source(path.read_bytes(), path)
        return run_tests(root, tests, output_dir, fail_fast=True, timeout=timeout)
    finally:
        path.write_bytes(raw)
        if path.read_bytes() != raw:
            raise HarnessError(f"source restoration failed: {path}")


def main():
    # Check every anchor and the syntax of every proposed mutant before any source write.
    originals = {}
    for mutant in MUTANTS:
        path, raw, changed = replacement(ROOT, mutant)
        originals[path] = raw
        compile_source(changed, path)
    started = time.monotonic()
    escaped = 0
    try:
        with tempfile.TemporaryDirectory(prefix="evalab-mutation-") as scratch:
            output_dir = Path(scratch)
            base = run_tests(ROOT, TESTS, output_dir / "baseline")
            if base["killed"]:
                raise HarnessError("unmodified tests fail; fix the baseline first:\n" + "\n".join(base["failures"]))
            print(f"baseline: {base['tests']} cases, {base['skipped']} skipped, "
                  f"{time.monotonic() - started:.1f}s", flush=True)
            for i, mutant in enumerate(MUTANTS, 1):
                result = run_mutant(ROOT, TESTS, mutant, output_dir / f"mutant-{i:03d}")
                escaped += not result["killed"]
                evidence = result["failures"][0] if result["killed"] else "no test failed"
                anchor = " ".join(mutant[1].split())[:70]
                print(f"{'killed ' if result['killed'] else 'ESCAPED'} [{i}/{len(MUTANTS)}] "
                      f"{mutant[0]}: {anchor} -- {evidence}", flush=True)
                if not result["killed"]:
                    raise HarnessError(f"mutant {i}/{len(MUTANTS)} escaped; stopped for review after restoring source")
    finally:
        unexpected = [path for path, raw in originals.items() if not path.exists() or path.read_bytes() != raw]
        for path in unexpected:
            path.write_bytes(originals[path])
        if unexpected:
            raise HarnessError("unexpected source changes restored: " + ", ".join(map(str, unexpected)))
    print(f"{len(MUTANTS) - escaped}/{len(MUTANTS)} mutants killed "
          f"({time.monotonic() - started:.1f}s); all source bytes restored", flush=True)
    return 1 if escaped else 0


def self_test():
    """Exercise failure classification/restoration without importing product code."""
    import py_compile
    import unittest

    class HarnessTests(unittest.TestCase):
        def setUp(self):
            self.scratch = tempfile.TemporaryDirectory(prefix="evalab-harness-test-")
            self.addCleanup(self.scratch.cleanup)
            self.root = Path(self.scratch.name)
            self.source = self.root / "subject.py"
            self.original = b"VALUE = 1\n"
            self.source.write_bytes(self.original)
            self.tests = self.root / "test_subject.py"
            self.tests.write_text("from subject import VALUE\ndef test_value():\n    assert VALUE == 1\n")

        def mutate(self, new, **kwargs):
            try:
                return run_mutant(self.root, self.tests, ("subject.py", "VALUE = 1", new),
                                  self.root / "run", **kwargs)
            finally:
                self.assertEqual(self.source.read_bytes(), self.original)

        def test_assertion_failure_ignores_stale_bytecode(self):
            py_compile.compile(str(self.source), doraise=True,
                               invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH)
            result = self.mutate("VALUE = 2")
            self.assertTrue(result["killed"])
            self.assertIn("test_value", result["failures"][0])

        def test_survivor_is_not_a_kill(self):
            self.assertFalse(self.mutate("VALUE = 1 # unchanged behavior")["killed"])

        def test_compile_failure_restores_source(self):
            with self.assertRaisesRegex(HarnessError, "does not compile"):
                self.mutate("VALUE = (")

        def test_collection_failure_is_not_a_kill(self):
            with self.assertRaisesRegex(HarnessError, "not a kill"):
                self.mutate("raise ImportError('synthetic collection failure')")

        def test_setup_failure_is_not_a_kill(self):
            self.tests.write_text("import pytest\n@pytest.fixture(autouse=True)\n"
                                  "def broken():\n    raise RuntimeError('synthetic fixture failure')\n"
                                  "def test_value():\n    assert False\n")
            with self.assertRaisesRegex(HarnessError, "not a kill"):
                self.mutate("VALUE = 2")

        def test_timeout_is_not_a_kill_and_restores_source(self):
            with self.assertRaisesRegex(HarnessError, "timed out"):
                self.mutate("import time; time.sleep(10); VALUE = 1", timeout=1)

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(HarnessTests))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        sys.exit(self_test())
    if sys.argv[1:]:
        sys.exit("usage: python tests/mutate.py [--self-test]")
    try:
        sys.exit(main())
    except HarnessError as exc:
        sys.exit(f"MUTATION ERROR: {exc}")
