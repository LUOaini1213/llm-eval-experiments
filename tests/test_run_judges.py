"""Runner persistence regressions. All models and files are local test doubles."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from evalab.items import load_jsonl, write_jsonl
from evalab.llm import Reply


@pytest.fixture
def runner(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / "scripts" / "e1_run_judges.py"
    spec = importlib.util.spec_from_file_location("isolated_judge_runner", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    monkeypatch.setattr(mod, "PREREG", tmp_path / "docs" / "PREREGISTRATION.md")
    monkeypatch.setattr(mod, "JUDGES", ["qwen2.5:3b", "llama3.2:3b"])
    mod.PREREG.parent.mkdir()
    mod.PREREG.write_bytes(b"Original plan\nMain-set size: 2\n")
    data = tmp_path / "data" / "e1"
    items = [{"item_id": f"item-{n}", "qid": f"q-{n}", "question": "2 + 2?",
              "reference": "4", "candidate": "4", "correct": True} for n in range(2)]
    for name in ("items_pilot", "items_main_order", "items_robust"):
        write_jsonl(data / f"{name}.jsonl", items)
    write_jsonl(data / "pairs.jsonl", [{"pair_id": "pair-0", "question": "2 + 2?",
                                      "reference": "4", "first": "4", "second": "5"}])
    (tmp_path / "results" / "e1").mkdir(parents=True)
    monkeypatch.setattr(mod, "model_digest", lambda model: f"digest-{model}")
    monkeypatch.setattr(mod, "unload", lambda model: None)
    mod.calls = []

    def fake_chat(model, messages, **kwargs):
        mod.calls.append((model, messages))
        text = "5" if "Rate the candidate" in messages[0]["content"] else "CORRECT"
        if "Reply with one letter" in messages[0]["content"]:
            text = "A"
        return Reply(text, model, 10, 1, 0.1)

    monkeypatch.setattr(mod, "chat", fake_chat)
    return mod


def run(runner, monkeypatch, which="pilot", judges=None, run_id=None):
    args = ["e1_run_judges.py", which]
    if judges is not None:
        args.append(",".join(judges))
    if run_id is not None:
        args.extend(["--run-id", run_id])
    monkeypatch.setattr(sys, "argv", args)
    runner.main()


def seed_legacy(runner, which="pilot"):
    rows = []
    for judge in runner.JUDGES:
        for cond in runner.CONDITIONS[which]:
            for item in runner.load(which):
                rows.append({"set": which, "item_id": item["item_id"], "judge": judge,
                             "condition": cond, "reply": "5" if cond == "score_ref" else "CORRECT",
                             "parsed": 5 if cond == "score_ref" else True,
                             "prompt_tokens": 10, "output_tokens": 1, "seconds": 0.1})
    path = runner.ROOT / "results" / "e1" / f"judgments_{which}.jsonl"
    write_jsonl(path, rows)
    if which == "main":
        manifest = {"started": "2026-09-28T13:47:31+0800", "n_items": 2,
                    "preregistration_sha256": hashlib.sha256(runner.PREREG.read_bytes()).hexdigest(),
                    "judges": {model: runner.model_digest(model) for model in runner.JUDGES}}
        (path.parent / "run_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return path


def test_single_judge_rerun_preserves_legacy_results_without_calls(runner, monkeypatch):
    path = seed_legacy(runner)
    before = path.read_bytes()
    run(runner, monkeypatch, judges=runner.JUDGES[:1])
    assert path.read_bytes() == before
    assert not runner.calls


def test_main_rerun_preserves_original_manifest_after_amendments(runner, monkeypatch):
    path = seed_legacy(runner, "main")
    manifest = path.parent / "run_manifest.json"
    before = manifest.read_bytes()
    runner.PREREG.write_bytes(runner.PREREG.read_bytes() + b"\n## Amendments\nExploratory analysis\n")
    run(runner, monkeypatch, "main", runner.JUDGES[:1])
    assert manifest.read_bytes() == before
    assert not runner.calls


def test_incomplete_legacy_run_refuses_to_guess_identity(runner, monkeypatch):
    path = seed_legacy(runner)
    write_jsonl(path, load_jsonl(path)[1:])
    before = path.read_bytes()
    with pytest.raises((ValueError, SystemExit), match="legacy|provenance|identity"):
        run(runner, monkeypatch, judges=runner.JUDGES[:1])
    assert path.read_bytes() == before
    assert not runner.calls


def test_main_preregistration_change_is_rejected_without_writes(runner, monkeypatch):
    path = seed_legacy(runner, "main")
    before = path.read_bytes(), (path.parent / "run_manifest.json").read_bytes()
    runner.PREREG.write_bytes(b"Changed plan\nMain-set size: 2\n")
    with pytest.raises((ValueError, SystemExit), match="preregistration|pre-registration"):
        run(runner, monkeypatch, "main", runner.JUDGES[:1])
    assert before == (path.read_bytes(), (path.parent / "run_manifest.json").read_bytes())
    assert not runner.calls


def test_new_run_merges_judges_and_only_calls_missing_cells(runner, monkeypatch):
    run(runner, monkeypatch, judges=runner.JUDGES[:1])
    path = runner.ROOT / "results" / "e1" / "judgments_pilot.jsonl"
    first = load_jsonl(path)
    assert len(first) == 8
    run(runner, monkeypatch, judges=runner.JUDGES[1:])
    assert load_jsonl(path)[:len(first)] == first
    assert len(load_jsonl(path)) == 16
    runner.calls.clear()
    before = path.read_bytes()
    run(runner, monkeypatch)
    assert path.read_bytes() == before
    assert not runner.calls


def test_interruption_checkpoints_successful_replies_and_resumes(runner, monkeypatch):
    fake_chat = runner.chat

    def failing_chat(*args, **kwargs):
        if len(runner.calls) == 2:
            raise RuntimeError("interrupted test model")
        return fake_chat(*args, **kwargs)

    monkeypatch.setattr(runner, "chat", failing_chat)
    with pytest.raises(RuntimeError, match="interrupted"):
        run(runner, monkeypatch, "main", runner.JUDGES[:1])
    path = runner.ROOT / "results" / "e1" / "judgments_main.jsonl"
    saved = load_jsonl(path)
    assert len(saved) == 2
    manifest = (path.parent / "run_manifest.json").read_bytes()
    monkeypatch.setattr(runner, "chat", fake_chat)
    runner.calls.clear()
    run(runner, monkeypatch, "main", runner.JUDGES[:1])
    assert len(runner.calls) == 6
    assert load_jsonl(path)[:2] == saved
    assert (path.parent / "run_manifest.json").read_bytes() == manifest


@pytest.mark.parametrize("change", ["item", "prompt", "options", "digest"])
def test_resume_rejects_changed_experiment(runner, monkeypatch, change):
    run(runner, monkeypatch, judges=runner.JUDGES[:1])
    path = runner.ROOT / "results" / "e1" / "judgments_pilot.jsonl"
    before = path.read_bytes()
    runner.calls.clear()
    if change == "item":
        source = runner.ROOT / "data" / "e1" / "items_pilot.jsonl"
        items = load_jsonl(source)
        items[0]["candidate"] = "5"
        write_jsonl(source, items)
    elif change == "prompt":
        monkeypatch.setattr(runner.J, "SYSTEM", "A new instruction.")
    elif change == "options":
        monkeypatch.setitem(runner.NUM_PREDICT, "bin_ref", 99)
    else:
        monkeypatch.setattr(runner, "model_digest", lambda model: "replacement-weights")
    with pytest.raises((ValueError, SystemExit), match="identity|configuration|digest"):
        run(runner, monkeypatch, judges=runner.JUDGES[1:])
    assert path.read_bytes() == before
    assert not runner.calls


def test_named_run_does_not_change_legacy_files(runner, monkeypatch):
    path = seed_legacy(runner, "main")
    before = path.read_bytes(), (path.parent / "run_manifest.json").read_bytes()
    run(runner, monkeypatch, "main", runner.JUDGES[:1], run_id="replication-1")
    fresh = path.parent / "runs" / "replication-1"
    assert len(load_jsonl(fresh / "judgments_main.jsonl")) == 8
    assert (fresh / "run_manifest.json").exists()
    assert before == (path.read_bytes(), (path.parent / "run_manifest.json").read_bytes())


@pytest.mark.parametrize("mutation", ["duplicate", "unknown_item", "wrong_set", "extra_order"])
def test_bad_existing_rows_fail_before_calls_or_writes(runner, monkeypatch, mutation):
    path = seed_legacy(runner)
    rows = load_jsonl(path)
    if mutation == "duplicate":
        rows.append(rows[0].copy())
    elif mutation == "unknown_item":
        rows[0]["item_id"] = "foreign-item"
    elif mutation == "wrong_set":
        rows[0]["set"] = "main"
    else:
        rows[0]["order"] = "AB"
    write_jsonl(path, rows)
    before = path.read_bytes()
    with pytest.raises((ValueError, SystemExit)):
        run(runner, monkeypatch)
    assert path.read_bytes() == before
    assert not runner.calls


def test_pairs_resume_keeps_both_presentation_orders(runner, monkeypatch):
    run(runner, monkeypatch, "pairs", runner.JUDGES[:1])
    run(runner, monkeypatch, "pairs", runner.JUDGES[1:])
    path = runner.ROOT / "results" / "e1" / "judgments_pairs.jsonl"
    rows = load_jsonl(path)
    assert len(rows) == 8
    assert {(row["judge"], row["condition"], row["order"]) for row in rows} == {
        (judge, condition, order) for judge in runner.JUDGES
        for condition in runner.CONDITIONS["pairs"] for order in ("AB", "BA")}


@pytest.mark.parametrize("value", ["0", "-1", "2.5", "100"])
def test_main_size_must_be_positive_integer_and_available(runner, monkeypatch, value):
    runner.PREREG.write_text(f"Main-set size: {value}\n", encoding="utf-8")
    with pytest.raises((ValueError, SystemExit)):
        run(runner, monkeypatch, "main")
    assert not runner.calls
    assert not (runner.ROOT / "results" / "e1" / "run_manifest.json").exists()


def test_failed_atomic_replace_preserves_prior_results(runner, monkeypatch):
    run(runner, monkeypatch, judges=runner.JUDGES[:1])
    path = runner.ROOT / "results" / "e1" / "judgments_pilot.jsonl"
    before = path.read_bytes()

    def cannot_replace(*args):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(runner.os, "replace", cannot_replace)
    with pytest.raises(OSError, match="disk failure"):
        run(runner, monkeypatch, judges=runner.JUDGES[1:])
    assert path.read_bytes() == before
    assert not list(path.parent.glob("*.tmp"))
    assert not list(path.parent.glob("*.lock"))


def test_existing_lock_prevents_competing_writer(runner, monkeypatch):
    path = seed_legacy(runner)
    lock = path.parent / ".judgments_pilot.lock"
    lock.write_text("another-process", encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises((ValueError, SystemExit), match="locked"):
        run(runner, monkeypatch)
    assert lock.read_text(encoding="utf-8") == "another-process"
    assert path.read_bytes() == before
    assert not runner.calls


def test_new_run_without_model_identity_does_not_write_manifest(runner, monkeypatch):
    monkeypatch.setattr(runner, "model_digest", lambda model: None)
    with pytest.raises((ValueError, SystemExit), match="digest"):
        run(runner, monkeypatch, "main")
    assert not list((runner.ROOT / "results" / "e1").glob("*.json*"))
    assert not runner.calls


def test_unknown_or_duplicate_judge_is_rejected_before_writes(runner, monkeypatch):
    for judges in (["not-in-design"], [runner.JUDGES[0]] * 2, [""]):
        with pytest.raises((ValueError, SystemExit), match="judges"):
            run(runner, monkeypatch, judges=judges)
    assert not runner.calls
    assert not list((runner.ROOT / "results" / "e1").iterdir())


def test_named_runs_do_not_share_a_cache_even_when_started_together(runner, monkeypatch):
    fake_chat = runner.chat
    cache_paths = []

    def record_cache(*args, **kwargs):
        cache_paths.append(kwargs["cache"].path)
        return fake_chat(*args, **kwargs)

    monkeypatch.setattr(runner, "chat", record_cache)
    monkeypatch.setattr(runner.time, "strftime", lambda *args: "2026-10-03T00:00:00+0800")
    run(runner, monkeypatch, judges=runner.JUDGES[:1], run_id="one")
    run(runner, monkeypatch, judges=runner.JUDGES[:1], run_id="two")
    assert len(set(cache_paths)) == 2


@pytest.mark.parametrize("raw", ["null", "[]", "false", '{"schema_version": true}'])
def test_invalid_existing_manifest_is_never_replaced(runner, monkeypatch, raw):
    manifest = runner.ROOT / "results" / "e1" / "run_manifest_pilot.json"
    manifest.write_text(raw, encoding="utf-8")
    before = manifest.read_bytes()
    with pytest.raises((ValueError, SystemExit), match="manifest"):
        run(runner, monkeypatch)
    assert manifest.read_bytes() == before
    assert not runner.calls


@pytest.mark.parametrize("run_id", ["../elsewhere", "..", "a/b", "a\\b", "", "a" * 65])
def test_run_id_stays_inside_dedicated_directory(runner, monkeypatch, run_id):
    with pytest.raises((ValueError, SystemExit), match="run-id"):
        run(runner, monkeypatch, run_id=run_id)
    assert not runner.calls
