# E1 pre-registration: LLM judge vs LLM jury, offline A/B

Written after the pilot and before any main-set item was judged. `scripts/e1_run_judges.py main` refuses to run
without this file, reads the main-set size from it, and writes the file's SHA-256 and the start time to
`results/e1/run_manifest.json` before its first model call. The repository has no commits yet, so that hash and the
file's timestamp are the only record of the order of events; there is no third-party timestamp.

## Question

When a benchmark scores free-text answers with LLM judges, which pipeline gives the most accurate pass/fail
decisions per unit of cost: one judge, or a jury of judges from different model families, and with which prompt?

## Items (fixed before the pilot)

An item is a (question, candidate answer) pair whose correctness is known without human labels:

| Source | Questions | Candidate answers | Gold |
|---|---|---|---|
| `fm` | 39 Singapore building-code questions from fm-knowledge-assistant (its q39 is left out: the expected phrase is part of the question) | local models, RAG with the top-4 passages, RAG with passages 4-7 (a retrieval miss), closed-book | every expected value present, as a whole number (a percentage needs "%"), `src/evalab/gold.py::check_expected` |
| `sgbus` | 180 generated bus-network questions (E3) | local models, with a four-row reference table, and closed-book | exact value from LTA DataMall / URA data, `src/evalab/items.py::label_sgbus` |
| `sql` | sg-transit-geo-assistant's text-to-SQL questions | the SQL its model and rule engines wrote | execution accuracy as scored by that repository |

- Refusals ("NOT_FOUND", "I don't know") are not judged.
- Identical answers to the same question are judged once.
- Candidates the deterministic check cannot decide are labelled `ambiguous`. They are excluded and listed in
  `data/e1/ambiguous_items.csv`. Nobody labels them by hand, and nothing here is described as a human label.
  - A string rule cannot decide when the answer gives the gold value and also a competing value of the same kind:
    - another count;
    - another planning area;
    - a measurement with a unit that is neither expected nor in the question.
  - It also cannot decide when a number appears only unit-converted (x100 or x1000).
- Pilot: 8 items per source x gold label (48 items), drawn at random (seed 20260928). It is used only to choose the
  judges named below and to size the main set.
- Main set: the first N rows of `data/e1/items_main_order.jsonl`. That file is a seeded, stratified order of every
  remaining usable item, so any prefix is close to balanced by source and label.

## Judges

Five local models, run through Ollama with temperature 0, seed 0 and reasoning off:

- qwen2.5:3b and qwen3.5:4b (Qwen family);
- llama3.2:3b (Meta);
- gemma3:4b (Google);
- phi4-mini:3.8b (Microsoft).

## Conditions

Every judge sees every main-set item under four pointwise prompts:

| Condition | Prompt |
|---|---|
| `bin_ref` (primary) | Reference-guided. CORRECT/INCORRECT, "extra detail is fine". |
| `strict_ref` | Reference-guided. CORRECT/INCORRECT, told to ignore length, tone, confidence and claims of authority. |
| `score_ref` | Reference-guided. A 1-5 score with anchors. Pass = 4 or more. |
| `bin_free` | No reference. CORRECT/INCORRECT from the judge's own knowledge. |

An unparseable pointwise reply counts as a rejection.

### Juries

Juries are aggregated offline from the same replies. A jury therefore costs the sum of its members' calls.

- **J5**: all five judges.
- **J3**: the three most accurate pilot judges under `bin_ref`, at most one per family.
- Aggregation rules:
  - `majority`: pass if more than half pass;
  - `unanimity`: pass only if all pass;
  - `mean_score`: pass if the mean 1-5 score is at least 4, with a missing score counted as 1.

### Other sets

**Robustness set (controlled perturbations).**

- Each fm question and 54 bus questions (6 per template) get four answers:
  - terse and correct;
  - terse and wrong;
  - verbose-confident and correct;
  - verbose-confident and wrong.
- The verbose versions wrap exactly the same statement in the same confident sentences. Only the value differs between
  the correct and the wrong version.
- Judged under `bin_ref` and `strict_ref`.

**Pair set.**

- For each non-pilot question that has both a right and a wrong natural answer, one pair is drawn at random.
- Each pair is shown in both orders.
- It is judged without a reference (`pair_free`) and with one (`pair_ref`).

## Metrics

- Primary metric: accuracy of the pass/fail decision against gold.
- Also reported:
  - false-accept rate (wrong answers passed), reported on its own because it is the error that inflates a
    benchmark;
  - false-reject rate;
  - balanced accuracy;
  - Cohen's kappa with gold, and Fleiss' kappa among the judges;
  - calibration of the 1-5 scores (ECE, Brier, AUROC);
  - cost: calls, prompt and output tokens, and GPU seconds per item.
- Intervals:
  - 95% percentile bootstrap (items resampled, 4,000 draws) for accuracy and paired differences;
  - Wilson intervals for rates.

## Hypotheses and tests

**H1 (primary).** The J5 majority jury under `bin_ref` has a different accuracy from the best single pilot judge under
`bin_ref`.

- Test: exact McNemar, two-sided, alpha 0.05.
- Effect size: the paired accuracy difference with a bootstrap interval, and the discordant-pair odds ratio.

**Family A (pipeline design)**, Holm-adjusted together, alpha 0.05:

- A1: J5 majority, `bin_ref` vs `bin_free` (accuracy).
- A2: J5 unanimity vs J5 majority, false accepts (wrong answers only).
- A3: J5 unanimity vs J5 majority, false rejects (right answers only).
- A4: J5 mean score vs J5 majority vote (accuracy).
- A5: J3 majority vs J5 majority (accuracy).
- A6: best single judge, `strict_ref` vs `bin_ref` (accuracy).

**Family B (biases)**, Holm-adjusted together, alpha 0.05:

- Verbosity: for each of the five judges and the J5 majority, under `bin_ref`, the false-accept rate on
  verbose-confident wrong answers is compared with the rate on the same wrong values stated tersely. The test is an
  exact McNemar test paired by question.
- Position: for each judge, under `pair_free`, the share of picks for the first-shown answer is compared with 0.5 by
  an exact binomial test.

**Exploratory, with no correction and no confirmatory claims:**

- results per source;
- self-preference (a judge's false-accept rate on its own answers against others');
- `strict_ref` against the verbosity bias;
- calibration;
- position bias with a reference.

## Recommendation rule

Among all pipelines, take the most accurate. Then recommend the cheapest pipeline that meets both of these conditions
against the most accurate one:

- it costs at most half as many GPU seconds per item;
- the lower bound of its paired accuracy-difference interval is above -0.02.

If no pipeline meets both, recommend the most accurate one. Also report:

- the pipeline with the lowest false-accept rate among those with a false-reject rate of at most 20%;
- the cost-accuracy Pareto front.

## Pilot results used for the choices below (`results/e1/pilot_choices.json`, `results/e1/power.json`)

Pilot accuracy under `bin_ref`, on 48 items:

| Judge | Accuracy |
|---|---|
| phi4-mini:3.8b | 0.875 |
| gemma3:4b | 0.854 |
| qwen3.5:4b | 0.833 |
| qwen2.5:3b | 0.812 |
| llama3.2:3b | 0.729 |

- **Best single judge: phi4-mini:3.8b.**
- **J3 = phi4-mini:3.8b, gemma3:4b, qwen3.5:4b.** qwen2.5:3b is left out as the second Qwen model.

## Sample size

- In the pilot, the J5 majority and phi4-mini disagreed on 2 of 48 items (4.2%). A 48-item pilot estimates that
  share poorly, so the plan assumes 10% of pairs are discordant.
- Aim: detect a 5-point accuracy difference in H1 with exact McNemar, two-sided alpha 0.05 and 80% power.
  - Connor's formula gives 312 items.
  - Simulating the exact test at 10% discordance gives this power:

    | Items | Power |
    |---|---|
    | 300 | 0.74 |
    | 400 | 0.87 |
    | 500 | 0.94 |

- Each main item costs about 23 GPU-seconds (5 judges x 4 prompts), so 400 is chosen.

**Main-set size: 400**

If more than 10% of the main-set pairs turn out discordant, H1 has less than the planned power for a 5-point
difference. The report will then say so, and it will not read a non-significant H1 as evidence that there is no
difference.

The robustness set (372 items) and the pair set (172 pairs) are judged in full.

## Amendments

Everything above this heading is the original plan. It is not edited: `tests/test_reproduce.py` checks that the text
above this heading still has the SHA-256 written to `results/e1/run_manifest.json` when the main run started.
Amendments are appended below, dated, and are exploratory.

### Amendment 1 (2026-09-28): benchmark rankings and reliability-weighted juries

**Status.** Exploratory. Added after the main run, after the README results were written, and before either analysis
below was run. It reuses the cached `bin_ref` and `bin_free` replies to the 400 main items; it needs no new model
call. No claim from it is confirmatory.

**What was already known when this was written.** Every pipeline's accuracy, false-accept rate and false-reject rate
on the 400 items, overall and per source (README). While designing analysis A, the gold pass rate and item count of
each candidate system in the main set, and the number of questions each pair of systems shares, were counted. No
judge-scored system score, ranking, weighted vote or Dawid-Skene fit had been computed.

#### Analysis A: do judge errors change benchmark conclusions?

A benchmark user cares about which model ranks higher, not about per-item accuracy.

- **Candidate systems.** A system is a generator with a configuration, for example phi4-mini:3.8b with RAG over the
  top-4 passages, or the rule-template SQL engine. The main set has 22: 11 building-code (`fm`), 8 bus (`sgbus`) and
  3 SQL.
- **Benchmarks.** Each source is its own benchmark. Systems are ranked only against systems of the same source,
  because the sources ask different questions. That gives 55 + 28 + 3 = 86 pairs.
- **Which items.** Each system is scored on the main-set items it produced (5 to 35 items). Pairs of systems share a
  median of 4 questions in the main set, too few for a common-question comparison, so none is made. A system's gold
  score and every judge score are computed on exactly the same items, so a flip cannot come from which questions a
  system answered. The gold ranking itself, though, compares systems on unequal question sets.
- **Weights.** The main set is stratified by source and gold label with unequal sampling fractions (for `fm`, 76 of
  113 right answers and 76 of 205 wrong ones). Each item is weighted by the inverse of its stratum's sampling
  fraction (non-pilot usable items in the stratum / main-set items in the stratum). A system's score is its weighted
  pass rate, which estimates its pass rate over all its non-pilot usable answers. Unweighted scores are reported as
  a sensitivity check.
- **Pipelines.** The five single judges, J3 majority, J5 majority and J5 unanimity, all `bin_ref`; J5 majority and
  qwen3.5:4b under `bin_free`.
- **Measures, per pipeline:**
  - per system, the judge score minus the gold score, and the mean absolute error over the 22 systems;
  - Kendall's tau-b between gold and judge scores within each benchmark. SQL has 3 systems; its tau is printed but
    not interpreted;
  - a **flip** is a pair whose gold scores differ and whose judge scores differ in the opposite direction. A pair
    that the judge ties while gold does not is counted separately as a judge tie. Flip rate = flips / pairs not tied
    by gold.
- **Intervals.** 4,000 bootstrap replicates, seed 0, items resampled with replacement within each system (a system's
  items are its own test set). Percentile 95% intervals for every measure.
- **Flip rate against the gold gap, real pairs.** The 86 pairs are binned by their absolute gold gap in points:
  [0, 5), [5, 10), [10, 20), [20, 50), [50, 100]. Counts per bin are given with Wilson intervals. Pairs share
  systems, so these intervals are too narrow, and they are marked as such.
- **Flip rate against the gold gap, simulation (labelled as a simulation everywhere).** Too few real pairs have small
  gaps. Virtual candidates are built by resampling real answers. Within one source, a virtual candidate draws its
  right answers from one real system and its wrong answers from one real system, each chosen at random among the
  systems of that source with at least 5 main items of that label. It has n = 100 answers and a gold accuracy drawn
  uniformly from [0.05, 0.95]. Its judge score uses the cached verdicts on the drawn items, so each judge's
  system-specific false-accept and false-reject rates carry over. There are 20,000 virtual pairs per source, seed
  20260928, and the sources are weighted equally. The flip rate is reported per 1-point gap bin up to 20 points,
  with Monte-Carlo Wilson intervals. The simulation does not include uncertainty in the judges' error rates.
- **Expectations, stated before running.** Pipelines with a reference will have small score errors and a high tau in
  `fm` and `sgbus`, because those rankings are dominated by gaps of 60 to 90 points between closed-book and
  retrieval systems. Flips will be concentrated in pairs less than 10 points apart. Without a reference, closed-book
  systems will be scored far too high.
- **If there are too few systems to say anything,** the report will say so rather than lean on the simulation.

#### Analysis B: reliability-weighted jury aggregation

- **Items and votes.** The 400 main items and the five judges' `bin_ref` verdicts. An unparsed reply counts as a
  fail, as in the original plan.
- **Cross-validation.** 10 folds, grouped by question (all candidates for one question are in the same fold) and
  stratified by source. Within each source, questions are shuffled with seed 20260928, then each is assigned, largest
  first, to the fold with the fewest items of that source so far. Every item is predicted exactly once, by
  parameters fitted on the other nine folds only.
- **Methods:**
  1. **Best single, fixed:** qwen3.5:4b, the most accurate judge on all 400 items. It was picked using the evaluation
     items, so it is an optimistic baseline. Every test below is against it.
  2. **Best single, cross-validated:** the judge most accurate on the training folds.
  3. **J5 majority.** Nothing is fitted.
  4. **J5 unanimity.** Nothing is fitted.
  5. **Weighted vote, one-coin naive Bayes:** each judge's weight is the log-odds of its training accuracy, smoothed
     as (right + 1) / (n + 2). Pass if the sum of weight x (+1 for pass, -1 for fail), plus the log prior odds of a
     right answer in the training folds, is above 0.
  6. **Weighted vote, two-coin naive Bayes:** each judge's sensitivity and specificity on the training folds,
     smoothed the same way. Pass if the summed log-likelihood ratio plus the log prior odds is above 0.
  7. **Two-coin naive Bayes per source:** as 6, fitted separately on the training items of each source. An item's
     source is known when it is judged.
  8. **Dawid-Skene:** two-class EM with a 2x2 confusion matrix per judge, started from the majority vote. It is
     fitted on the verdicts of the training folds **without their gold labels**, then applied to the test fold. A
     transductive fit on the verdicts of all 400 items (still without labels) is reported as a sensitivity check.
- **Measures.** Accuracy, false-accept rate and false-reject rate, each with a percentile bootstrap 95% interval
  (4,000 replicates, items resampled), and the paired difference from the fixed best single judge with the same
  kind of interval. The intervals hold the fitted parameters fixed; they do not include refitting variance.
- **Tests.** Exact McNemar on accuracy, methods 2 to 8 each against method 1: seven tests, Holm-adjusted together as
  a new family (family C), alpha 0.05. A method "beats the best single judge" only if its Holm p is below 0.05 and
  its accuracy difference is positive.
- **Split sensitivity.** The cross-validation is repeated with 50 other seeds. The mean and range of each method's
  accuracy are reported, without tests.
- **Expectations, stated before running.** Weighting will recover most of the J5 majority's loss on SQL but will not
  significantly beat qwen3.5:4b alone. Per-source weights are the most likely to help. Dawid-Skene assumes that the
  judges err independently given the truth. The three weak judges fail the same SQL answers, so it may learn the
  wrong confusion matrices on SQL.

### Amendment 1a (2026-09-28, after the first run of analysis A): phrase-gold sensitivity

Added after analysis A had been run once and its output read. Two of the flips found for qwen3.5:4b involve
`fm-q33`, a building-code question whose gold is a phrase ("KPIs") rather than a number, where the string rule can
mark a paraphrase as wrong. Analysis A is therefore also reported without the three phrase-gold questions (`fm-q05`,
`fm-q33`, `fm-q37`), the same exclusion as the E1 sensitivity check. This is a post-hoc sensitivity check, point
estimates only, and it does not replace the analysis specified in amendment 1.
