"""Post-hoc question-cluster sensitivity, keeping the original E1 item-weighted estimand and results."""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from . import analysis as A
from . import stats as S


def question_profile(items):
    """Validate question identity and describe repeated candidates, without dropping any answer."""
    if not items:
        raise ValueError("need at least one item")
    identities, question_ids, item_ids, counts = {}, {}, set(), Counter()
    for item in items:
        qid = item.get("qid")
        if not isinstance(qid, str) or not qid.strip():
            raise ValueError("each item needs a non-empty question ID")
        if item["item_id"] in item_ids:
            raise ValueError("duplicate item ID")
        item_ids.add(item["item_id"])
        identity = (item["source"], item["question"])
        if qid in identities and identities[qid] != identity:
            raise ValueError("a question ID must have one source and question text")
        if identity in question_ids and question_ids[identity] != qid:
            raise ValueError("identical source/question text cannot have multiple question IDs")
        identities[qid] = identity
        question_ids[identity] = qid
        counts[qid] += 1
    histogram = Counter(counts.values())
    return {"n_items": len(items), "n_questions": len(counts),
            "n_distinct_question_texts": len({it["question"] for it in items}),
            "repeated_questions": sum(n > 1 for n in counts.values()),
            "singleton_questions": histogram[1], "extra_candidates": len(items) - len(counts),
            "max_candidates_per_question": max(counts.values()),
            "candidate_count_histogram": {str(n): histogram[n] for n in sorted(histogram)},
            "question_candidate_counts": dict(sorted(counts.items()))}


def question_weighted_mean(values, qids):
    by_question = defaultdict(list)
    for qid, value in zip(qids, values):
        by_question[qid].append(value)
    return float(np.mean([np.mean(v) for v in by_question.values()]))


def mean_sensitivity(values, items, b=4000, seed=0):
    """Same item-weighted point estimate, with three explicitly different resampling designs."""
    qids = [it["qid"] for it in items]
    sources = [it["source"] for it in items]
    item_ci = list(S.bootstrap_ci(values, b=b, seed=seed))
    cluster_ci = list(S.cluster_bootstrap_ci(values, qids, b=b, seed=seed))
    stratified_ci = list(S.cluster_bootstrap_ci(values, qids, strata=sources, b=b, seed=seed))
    return {"n_items": len(items), "n_questions": len(set(qids)), "estimate": float(np.mean(values)),
            "item_bootstrap_ci": item_ci, "question_cluster_ci": cluster_ci,
            "source_stratified_question_ci": stratified_ci,
            "question_weighted_estimate": question_weighted_mean(values, qids),
            "ci_widths": {"item": item_ci[1] - item_ci[0], "question": cluster_ci[1] - cluster_ci[0],
                          "source_stratified_question": stratified_ci[1] - stratified_ci[0]}}


def build_report(items, pipelines, comparisons, b=4000, seed=0):
    """comparisons maps a label to (pipeline a, pipeline b, gold subset or None). No clustered p-values."""
    profile = question_profile(items)
    for pipeline in pipelines.values():
        if set(pipeline.decisions) != {it["item_id"] for it in items}:
            raise ValueError("every pipeline must cover exactly the declared items")
    metrics = {}
    for key, pipeline in pipelines.items():
        values = np.asarray(A.right(pipeline, items), dtype=float)
        metrics[key] = {"accuracy": mean_sensitivity(values, items, b, seed)}
        for gold, rate, transform in (("incorrect", "false_accept_rate", bool),
                                       ("correct", "false_reject_rate", lambda passed: not passed)):
            subset = [it for it in items if it["gold"] == gold]
            if not subset:
                continue
            errors = [float(transform(pipeline.decisions[it["item_id"]])) for it in subset]
            metrics[key][rate] = mean_sensitivity(errors, subset, b, seed)
            metrics[key][rate]["original_wilson_ci"] = list(S.wilson(int(sum(errors)), len(subset)))
    paired = {}
    for label, (a, b_key, gold) in comparisons.items():
        subset = [it for it in items if gold is None or it["gold"] == gold]
        if not subset:
            raise ValueError("comparison subset must be non-empty")
        xa, xb = np.asarray(A.right(pipelines[a], subset)), np.asarray(A.right(pipelines[b_key], subset))
        diff = xa.astype(float) - xb.astype(float)
        paired[label] = {"a": a, "b": b_key, "subset": gold or "all", **mean_sensitivity(diff, subset, b, seed)}
    sources = {source: question_profile([it for it in items if it["source"] == source])
               for source in sorted({it["source"] for it in items})}
    return {"design": {"status": "post-hoc sensitivity; original item-level analyses retained",
                       "bootstrap_replicates": b, "seed": seed, "alpha": 0.05,
                       "cluster": "qid (validated against source and exact question text)",
                       "strata": "source for the stratified variant; none for the pooled variant",
                       "estimand": "item-weighted mean; sampled cluster sums / sampled item counts",
                       "conditional_rates": "resample question clusters represented in the selected gold subset",
                       "question_weighted_estimate": "descriptive alternative: equal weight per question",
                       "inference": "question independence is an assumption, not a finding; no clustered p-values"},
            "sample": profile, "by_source": sources, "pipelines": metrics, "comparisons": paired}
