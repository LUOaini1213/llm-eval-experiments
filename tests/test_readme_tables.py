"""The README's generated blocks must match the committed result files, and every environment variable is documented."""
import copy
import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("readme_tables", ROOT / "scripts/readme_tables.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
README = (ROOT / "README.md").read_text(encoding="utf-8")


def test_every_generated_block_matches_the_result_files():
    assert R.stale_blocks(README) == []


def test_glance_table_is_on_the_first_screen_and_covers_every_part():
    start, end = R.markers("glance")
    head = README[:README.index(end)]
    assert start in head and head.count("\n") < 40
    block = head[head.index(start):]
    for part in ("| E1 |", "| E2 |", "| E3 |", "| E4 |", "| E5 |"):
        assert part in block


def test_a_changed_result_number_is_caught(monkeypatch):
    real = R.load

    def load(p):
        data = real(p)
        if Path(p).name == "main_metrics.json":
            data = copy.deepcopy(data)
            data["qwen3.5:4b|bin_ref"]["accuracy"] += 0.001
        return data

    monkeypatch.setattr(R, "load", load)
    assert set(R.stale_blocks(README)) == {"glance", "e1-main"}


def test_a_changed_report_table_is_caught(monkeypatch):
    real = R.first_table

    def first_table(p):
        table = real(p)
        return table.replace("3/27 = 11.1%", "2/27 = 7.4%") if p.name == "report.md" else table

    monkeypatch.setattr(R, "first_table", first_table)
    assert R.stale_blocks(README) == ["e4-thresholds"]


@pytest.mark.parametrize("defect", ["missing", "duplicate"])
def test_missing_or_duplicate_markers_are_refused(defect):
    if defect == "missing":
        text, name = README.replace("<!-- generated:e1-main:end -->", ""), "e1-main"
    else:
        text, name = README + "\n<!-- generated:glance:start -->\n", "glance"
    with pytest.raises(ValueError, match=name):
        R.update_readme(text)


def test_every_environment_variable_is_documented():
    pattern = re.compile(r"""os\.environ(?:\.get|\.setdefault)?\s*[\(\[]\s*["']([A-Z0-9_]+)["']""")
    names = set()
    for folder in ("scripts", "src", "moonshot_ext"):
        for path in (ROOT / folder).rglob("*.py"):
            names |= set(pattern.findall(path.read_text(encoding="utf-8")))
    assert {"SG_TRANSIT_ROOT", "SG_BUS_RAW", "FMKA_ROOT", "GEO_ROOT", "MOONSHOT_DATA"} <= names
    table = README[README.index("**Environment variables.**"):README.index("**Raw bus data.**")]
    missing = sorted(n for n in names if f"`{n}`" not in table)
    assert missing == []
