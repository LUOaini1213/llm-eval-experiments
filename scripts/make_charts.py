"""Draw the README charts (docs/*.png) from the result files. No model needed.

Colours: the first three categorical slots of the reference palette (blue, orange, aqua), which pass the
colour-vision-deficiency checks as a set; greys for context. Every series is also labelled directly.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R1, R2, R3 = (ROOT / "results" / x for x in ("e1", "e2", "e3"))
DOCS = ROOT / "docs"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
JUDGES = ["qwen2.5:3b", "llama3.2:3b", "phi4-mini:3.8b", "gemma3:4b", "qwen3.5:4b"]
COND_LABEL = {"bin_ref": "with reference", "strict_ref": "strict rubric", "score_ref": "1-5 score",
              "bin_free": "no reference"}


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def load(p):
    return json.loads(p.read_text(encoding="utf-8"))


def short(name):
    j, cond = name.split("|")
    return f"{j}, {COND_LABEL.get(cond, cond)}"


def tradeoff():
    m = load(R1 / "main_metrics.json")
    rec = load(R1 / "recommendation.json")
    fig, ax = plt.subplots(figsize=(8.6, 5.6), dpi=150)
    fig.patch.set_facecolor(SURF)
    style(ax)
    groups = [("single judge, with reference", lambda k: "J" not in k.split("|")[0][:1] and k.endswith("|bin_ref"), BLUE),
              ("single judge, no reference", lambda k: not k.startswith("J") and k.endswith("|bin_free"), ORANGE),
              ("jury (3 or 5 judges)", lambda k: k.startswith("J"), AQUA)]
    shown = set()
    for label, sel, col in groups:
        ks = [k for k in m if sel(k)]
        shown |= set(ks)
        ax.scatter([m[k]["seconds_per_item"] for k in ks], [m[k]["accuracy"] for k in ks], s=46, color=col,
                   edgecolor=SURF, linewidth=1.5, label=label, zorder=3)
    rest = [k for k in m if k not in shown]
    ax.scatter([m[k]["seconds_per_item"] for k in rest], [m[k]["accuracy"] for k in rest], s=30, color="#b9b8b2",
               edgecolor=SURF, linewidth=1.2, label="single judge, other rubric", zorder=2)
    front = sorted((k for k in rec["pareto_front_seconds"]), key=lambda k: m[k]["seconds_per_item"])
    ax.plot([m[k]["seconds_per_item"] for k in front], [m[k]["accuracy"] for k in front], color=INK2, lw=1.2,
            ls="--", zorder=1, label="Pareto front")
    xmid = sorted(v["seconds_per_item"] for v in m.values())[len(m) // 2]
    for k in sorted({rec["recommended"], rec["most_accurate"]}):
        x, y = m[k]["seconds_per_item"], m[k]["accuracy"]
        tag = "recommended: " if k == rec["recommended"] else "most accurate: "
        left = x > xmid
        ax.annotate(tag + short(k), (x, y), xytext=(-8 if left else 8, 8), textcoords="offset points",
                    fontsize=8.5, color=INK, ha="right" if left else "left")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("GPU seconds per judged item (log scale; GTX 1650, 4 GB)", color=INK2, fontsize=9.5)
    ax.set_ylabel("Accuracy against deterministic gold", color=INK2, fontsize=9.5)
    ax.set_title(f"Accuracy against cost, {m['J5-majority|bin_ref']['n']} judged answers", color=INK, fontsize=11,
                 loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3)
    fig.tight_layout()
    fig.savefig(DOCS / "fig_tradeoff.png")
    plt.close(fig)


def error_rates():
    m = load(R1 / "main_metrics.json")
    ch = load(R1 / "pilot_choices.json")
    keys = [f"{j}|bin_ref" for j in JUDGES] + ["J3-majority|bin_ref", "J5-majority|bin_ref", "J5-unanimity|bin_ref",
                                               "J5-mean_score|score_ref", "J5-majority|bin_free"]
    labels = [short(k).replace(", with reference", "") for k in keys]
    labels = [lb.replace("J3-majority", f"J3 majority ({', '.join(x.split(':')[0] for x in ch['jury3'])})")
                .replace("J5-majority", "J5 majority").replace("J5-unanimity", "J5 unanimity")
                .replace("J5-mean_score, 1-5 score", "J5 mean score") for lb in labels]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.8), dpi=150, sharey=True)
    fig.patch.set_facecolor(SURF)
    y = list(range(len(keys)))[::-1]
    for ax, (what, lo_hi, col, title) in zip(axes, [
            ("false_accept_rate", "false_accept_ci", ORANGE, "False accepts: wrong answers passed"),
            ("false_reject_rate", "false_reject_ci", BLUE, "False rejects: right answers failed")]):
        style(ax)
        vals = [m[k][what] for k in keys]
        err = [[v - m[k][lo_hi][0] for k, v in zip(keys, vals)], [m[k][lo_hi][1] - v for k, v in zip(keys, vals)]]
        ax.errorbar(vals, y, xerr=err, fmt="o", color=col, ms=6, ecolor=col, elinewidth=1.4, capsize=0)
        for v, yy in zip(vals, y):
            ax.text(v, yy + 0.28, f"{v:.0%}", fontsize=7.5, color=INK2, ha="center")
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_title(title, color=INK, fontsize=10, loc="left")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=8.5, color=INK)
    n_wrong = m[keys[0]]["n_wrong_answers"]
    n_right = m[keys[0]]["n_right_answers"]
    fig.suptitle(f"Error rates with 95% Wilson intervals ({n_wrong} wrong and {n_right} right answers)", color=INK,
                 fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(DOCS / "fig_error_rates.png")
    plt.close(fig)


def biases():
    rob = load(R1 / "robustness.json")
    pw = load(R1 / "pairwise.json")
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6), dpi=150)
    fig.patch.set_facecolor(SURF)
    ax = axes[0]
    style(ax)
    names = JUDGES + ["J5-majority"]
    y = list(range(len(names)))[::-1]
    b = [rob[f"{j}|bin_ref"]["false_reject"]["verbose_rate"] for j in names]
    s = [rob[f"{j}|strict_ref"]["false_reject"]["verbose_rate"] for j in names]
    terse = max(rob[f"{j}|{c}"]["false_reject"]["terse_rate"] for j in names for c in ("bin_ref", "strict_ref"))
    for a, c, yy in zip(b, s, y):
        ax.plot([a, c], [yy, yy], color=GRID, lw=2, zorder=1)
    ax.scatter(b, y, color=BLUE, s=40, zorder=3, label="basic rubric")
    ax.scatter(s, y, color=ORANGE, s=40, zorder=3, label="strict rubric (\"ignore tone and confidence\")")
    ax.set_yticks(y)
    ax.set_yticklabels([n.replace("J5-majority", "J5 majority") for n in names], fontsize=8.5, color=INK)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Right answers in a verbose, confident style: share failed", color=INK, fontsize=10, loc="left")
    note = ("Stated tersely, none of these right answers was failed." if terse == 0 else
            f"Stated tersely, at most {terse:.0%} of them were failed.")
    ax.text(0.06, -0.34, note, fontsize=7.5, color=INK2, transform=ax.transAxes)
    ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.4, -0.1), ncol=1)
    ax = axes[1]
    style(ax)
    y = list(range(len(JUDGES)))[::-1]
    for cond, col, lab in (("pair_free", ORANGE, "no reference"), ("pair_ref", BLUE, "with reference")):
        ax.scatter([pw[f"{j}|{cond}"]["first_slot_share"] for j in JUDGES], y, color=col, s=40, zorder=3, label=lab)
    ax.axvline(0.5, color=INK2, lw=1, ls="--")
    ax.text(0.51, -0.35, "no position bias", fontsize=7.5, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels(JUDGES, fontsize=8.5, color=INK)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Position: share of picks for the first answer", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    fig.tight_layout()
    fig.savefig(DOCS / "fig_biases.png")
    plt.close(fig)


def flip_vs_gap():
    """Amendment 1, analysis A (exploratory): pairwise ranking flips against the gold score gap."""
    r = load(R1 / "ranking.json")
    show = [("qwen3.5:4b|bin_ref", "qwen3.5:4b", BLUE), ("J3-majority|bin_ref", "J3 majority", AQUA),
            ("J5-majority|bin_ref", "J5 majority", ORANGE)]
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.6), dpi=150, gridspec_kw={"width_ratios": [1, 1.35]})
    fig.patch.set_facecolor(SURF)
    ax = axes[0]
    style(ax)
    bins = r["pipelines"][show[0][0]]["real_pairs_by_gap"]
    xs = list(range(len(bins)))
    for off, (k, lab, col) in zip((-0.18, 0, 0.18), show):
        b = r["pipelines"][k]["real_pairs_by_gap"]
        ax.scatter([x + off for x in xs], [v["flips"] / v["pairs"] for v in b], color=col, s=40, zorder=3,
                   edgecolor=SURF, linewidth=1.2, label=lab)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{v['lo']}-{v['hi']}\n{v['pairs']} pairs" for v in bins], fontsize=8, color=INK2)
    ax.set_ylim(-0.03, 1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Gold score gap between the two systems (points)", color=INK2, fontsize=9)
    ax.set_ylabel("Pairs the judge orders the other way", color=INK2, fontsize=9)
    n_sys = r["design"]["n_systems"]
    ax.set_title(f"Real systems ({n_sys}, ranked within source)", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax = axes[1]
    style(ax)
    sim = r["simulation"]
    for k, lab, col in show + [("J5-majority|bin_free", "J5 majority, no reference", "#9a9993")]:
        c = sim["curves"][k]["by_gap"]
        g = [v["gap_points"] for v in c]
        ax.fill_between(g, [v["wilson"][0] for v in c], [v["wilson"][1] for v in c], color=col, alpha=0.15, lw=0)
        ax.plot(g, [v["flip_rate"] for v in c], color=col, lw=2, label=lab)
    ax.set_ylim(-0.03, 1)
    ax.set_xlim(1, max(g))
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Gold accuracy gap between two virtual candidates (points)", color=INK2, fontsize=9)
    ax.set_title(f"SIMULATION: virtual candidates, {sim['answers_per_candidate']} answers each", color=INK,
                 fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("How often a judge reverses the gold order of two systems (exploratory)", color=INK,
                 fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(DOCS / "fig_flip_vs_gap.png")
    plt.close(fig)


def aggregation():
    """Amendment 1, analysis B (exploratory): jury aggregation rules, out-of-fold, on the 400 main items."""
    a = load(R1 / "aggregation.json")["methods"]
    keys = ["best_single_fixed", "two_coin_per_source", "two_coin", "one_coin", "J3_majority", "J5_majority",
            "dawid_skene", "J5_unanimity"]
    labels = [a[k]["label"] for k in keys]
    y = list(range(len(keys)))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.4), dpi=150, sharey=True, gridspec_kw={"width_ratios": [1.1, 1]})
    fig.patch.set_facecolor(SURF)
    ax = axes[0]
    style(ax)
    acc = [a[k]["accuracy"] for k in keys]
    err = [[v - a[k]["accuracy_ci"][0] for k, v in zip(keys, acc)], [a[k]["accuracy_ci"][1] - v for k, v in zip(keys, acc)]]
    ax.errorbar(acc, y, xerr=err, fmt="o", color=BLUE, ms=6, ecolor=BLUE, elinewidth=1.4, capsize=0)
    ax.axvline(a["best_single_fixed"]["accuracy"], color=INK2, lw=1, ls="--")
    for k, v, yy in zip(keys, acc, y):
        ax.text(a[k]["accuracy_ci"][1] + 0.005, yy, f"{v:.1%}", fontsize=7.5, color=INK2, va="center")
    ax.set_xlim(0.7, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_title("Accuracy, 95% bootstrap interval", color=INK, fontsize=10, loc="left")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5, color=INK)
    ax = axes[1]
    style(ax)
    fa = [a[k]["false_accept_rate"] for k in keys]
    fr = [a[k]["false_reject_rate"] for k in keys]
    for u, v, yy in zip(fa, fr, y):
        ax.plot([u, v], [yy, yy], color=GRID, lw=2, zorder=1)
    ax.scatter(fa, y, color=ORANGE, s=40, zorder=3, label="false accepts (wrong answers passed)")
    ax.scatter(fr, y, color=AQUA, s=40, zorder=3, label="false rejects (right answers failed)")
    ax.set_xlim(0, 0.5)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Error rates", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=1)
    fig.suptitle("Jury aggregation, out-of-fold on 400 items (exploratory; dashed line: qwen3.5:4b alone)",
                 color=INK, fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(DOCS / "fig_aggregation.png")
    plt.close(fig)


def main():
    DOCS.mkdir(exist_ok=True)
    tradeoff()
    error_rates()
    biases()
    if (R1 / "ranking.json").exists():
        flip_vs_gap()
    if (R1 / "aggregation.json").exists():
        aggregation()
    print("charts written to", DOCS)


if __name__ == "__main__":
    main()
