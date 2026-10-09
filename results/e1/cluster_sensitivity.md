# E1 question-cluster sensitivity (2026-10-09)

Post-hoc analysis of the committed replies; no model calls or changes to frozen labels, replies,
pre-registration, original metrics, McNemar tests, Holm corrections or recommendation.

The 400 main items contain 188 distinct questions: 83 with one candidate and 105 with multiple candidates.
There are 212 additional candidates beyond one per question; the largest cluster has 7 candidates. Repeated questions are different candidate answers, not duplicate judgments.

| Source | Items | Questions | Questions with multiple candidates |
|---|---:|---:|---:|
| fm | 152 | 39 | 36 |
| sgbus | 152 | 101 | 35 |
| sql | 96 | 48 | 34 |

Candidate count per question: 1 candidates: 83 questions, 2 candidates: 50 questions, 3 candidates: 29 questions, 4 candidates: 10 questions, 5 candidates: 9 questions, 6 candidates: 4 questions, 7 candidates: 3 questions.

## Method and assumptions

All intervals use 4,000 percentile-bootstrap draws, seed 0, alpha 0.05. The item column reproduces
the original item-resampled interval. The pooled question column samples 188 questions with replacement,
taking every candidate of each selected question together. The stratified column samples the original
number of questions within each source. Both retain the item-weighted statistic: total successes
/ total sampled items (the denominator varies with cluster size). Paired differences resample both
pipelines together. Conditional false-accept/reject rates and gold-subset comparisons resample the
questions represented in that subset, keeping all its candidate answers together.

The question-weighted column is a separate descriptive estimand: average each question's candidate-level
accuracy, then give every question equal weight. It shows sensitivity to repeat-question weighting; it
does not replace the original accuracy or arbitrarily select one candidate per question.

Question resampling allows dependence within a question and assumes independent questions (within source
for the stratified design). Similar templates or shared entities across questions can still violate that
assumption. These synthetic/purposive questions are not a random sample of all Singapore questions.
Clustered intervals may widen or narrow; neither proves the original item-independence assumption false.
The original McNemar p-values remain item-level results; no new p-values, significance claims or power
claims are derived here. The analysis holds judge replies, model selection and labels fixed.

Method background: [Deen & de Rooij, ClusterBootstrap](https://doi.org/10.3758/s13428-019-01252-y).

## Accuracy sensitivity

Percentages and 95% intervals. Item-weighted accuracy is unchanged in every row.

| Pipeline | Item-weighted % | Original item CI | Pooled question CI | Source-stratified question CI | Equal-question % |
|---|---:|---|---|---|---:|
| qwen2.5:3b / bin_ref | 81.50 | [77.75, 85.25] | [76.71, 86.03] | [77.72, 85.05] | 82.59 |
| llama3.2:3b / bin_ref | 68.75 | [64.25, 73.00] | [62.83, 74.32] | [64.29, 72.79] | 67.81 |
| phi4-mini:3.8b / bin_ref | 84.50 | [81.00, 88.00] | [79.62, 89.07] | [80.57, 88.18] | 84.13 |
| gemma3:4b / bin_ref | 90.50 | [87.50, 93.25] | [87.23, 93.52] | [87.28, 93.44] | 89.89 |
| qwen3.5:4b / bin_ref | 93.25 | [90.75, 95.50] | [90.18, 96.07] | [90.09, 96.03] | 93.74 |
| qwen2.5:3b / strict_ref | 80.25 | [76.50, 84.00] | [75.56, 84.67] | [76.67, 83.70] | 82.31 |
| llama3.2:3b / strict_ref | 77.75 | [73.50, 81.51] | [72.10, 82.96] | [74.30, 80.98] | 76.51 |
| phi4-mini:3.8b / strict_ref | 82.75 | [79.00, 86.50] | [77.38, 87.73] | [78.72, 86.46] | 82.18 |
| gemma3:4b / strict_ref | 92.75 | [90.00, 95.25] | [89.83, 95.40] | [89.82, 95.44] | 92.96 |
| qwen3.5:4b / strict_ref | 92.25 | [89.50, 94.75] | [88.83, 95.30] | [88.98, 95.09] | 92.00 |
| qwen2.5:3b / score_ref | 65.25 | [60.50, 70.00] | [60.20, 70.00] | [61.03, 69.54] | 68.14 |
| llama3.2:3b / score_ref | 69.00 | [64.50, 73.50] | [63.85, 74.02] | [64.01, 73.67] | 71.19 |
| phi4-mini:3.8b / score_ref | 84.25 | [80.50, 87.50] | [79.95, 88.41] | [80.11, 88.18] | 85.59 |
| gemma3:4b / score_ref | 81.75 | [78.00, 85.50] | [77.98, 85.20] | [77.93, 85.39] | 81.33 |
| qwen3.5:4b / score_ref | 79.75 | [75.75, 83.50] | [75.50, 83.85] | [75.53, 83.82] | 78.56 |
| qwen2.5:3b / bin_free | 50.50 | [45.50, 55.25] | [45.27, 55.84] | [45.19, 55.70] | 53.13 |
| llama3.2:3b / bin_free | 44.00 | [39.25, 49.00] | [38.92, 49.16] | [39.12, 49.02] | 46.62 |
| phi4-mini:3.8b / bin_free | 54.75 | [49.75, 59.50] | [49.12, 60.30] | [49.21, 60.24] | 57.45 |
| gemma3:4b / bin_free | 57.00 | [52.00, 61.75] | [51.55, 62.47] | [51.93, 62.09] | 57.03 |
| qwen3.5:4b / bin_free | 60.00 | [55.00, 64.75] | [55.19, 64.75] | [55.50, 64.54] | 60.40 |
| J5-majority / bin_ref | 86.75 | [83.50, 90.00] | [82.62, 90.83] | [83.25, 90.00] | 85.30 |
| J5-unanimity / bin_ref | 75.00 | [70.75, 79.25] | [69.38, 80.29] | [71.46, 78.33] | 75.63 |
| J3-majority / bin_ref | 92.50 | [89.75, 95.00] | [89.60, 95.10] | [89.54, 95.07] | 92.38 |
| J3-unanimity / bin_ref | 85.50 | [82.00, 89.00] | [80.91, 89.92] | [81.98, 88.78] | 85.04 |
| J5-majority / strict_ref | 87.25 | [84.00, 90.50] | [82.95, 91.25] | [83.91, 90.39] | 86.21 |
| J5-unanimity / strict_ref | 74.75 | [70.50, 79.00] | [69.25, 79.85] | [71.35, 78.16] | 76.10 |
| J3-majority / strict_ref | 92.00 | [89.25, 94.50] | [88.58, 95.07] | [88.80, 94.85] | 91.68 |
| J3-unanimity / strict_ref | 83.75 | [80.25, 87.25] | [78.64, 88.55] | [80.15, 87.13] | 82.95 |
| J5-majority / bin_free | 54.25 | [49.25, 59.00] | [49.06, 59.69] | [48.89, 59.49] | 54.97 |
| J5-unanimity / bin_free | 45.25 | [40.50, 50.00] | [40.48, 50.25] | [40.72, 50.00] | 48.86 |
| J3-majority / bin_free | 57.75 | [52.75, 62.50] | [52.32, 63.20] | [52.55, 62.74] | 57.55 |
| J3-unanimity / bin_free | 54.00 | [49.25, 58.75] | [48.48, 59.45] | [48.48, 59.40] | 56.64 |
| J5-mean_score / score_ref | 81.50 | [77.75, 85.25] | [76.83, 86.02] | [78.15, 84.58] | 80.56 |
| J3-mean_score / score_ref | 87.50 | [84.25, 90.75] | [83.90, 90.78] | [83.90, 90.75] | 86.65 |

## Paired accuracy differences

Points, a minus b, with 95% intervals; for gold subsets this is classification correctness
(A2 is specificity gain, A3 is sensitivity change). Existing test names and contrasts are retained.

| Contrast | Items / questions | Original difference | Original item CI | Pooled question CI | Source-stratified question CI | Equal-question difference |
|---|---:|---:|---|---|---|---:|
| H1 | 400 / 188 | +2.25 | [0.00, 4.50] | [-0.25, 4.98] | [-0.25, 4.87] | +1.17 |
| A1 reference-guided vs reference-free (J5 majority) | 400 / 188 | +32.50 | [27.25, 38.00] | [26.34, 38.53] | [27.23, 37.80] | +30.33 |
| A2 unanimity vs majority, false accepts (J5) | 173 / 118 | +6.94 | [3.47, 10.98] | [2.96, 11.67] | [3.05, 11.49] | +6.21 |
| A3 unanimity vs majority, false rejects (J5) | 227 / 136 | -25.99 | [-31.72, -20.26] | [-32.65, -19.56] | [-32.19, -20.00] | -22.92 |
| A4 mean score vs majority vote (J5) | 400 / 188 | -5.25 | [-8.50, -2.25] | [-8.42, -2.10] | [-8.23, -2.26] | -4.74 |
| A5 three-judge vs five-judge majority | 400 / 188 | +5.75 | [3.00, 8.50] | [2.66, 9.09] | [3.12, 8.42] | +7.07 |
| A6 strict rubric vs basic rubric (best single judge) | 400 / 188 | -1.75 | [-3.25, -0.25] | [-3.64, 0.00] | [-3.43, -0.23] | -1.95 |

H1 remains an item-weighted difference of +2.25 points. Its pooled question CI is [-0.25, 4.98]; the source-stratified question CI is [-0.25, 4.87]. This is sensitivity evidence, not a replacement for the pre-registered test.

## Conditional error rates

The original error-rate intervals were Wilson intervals, not bootstrap intervals. They are preserved
and labelled accordingly; newly computed item and question bootstrap intervals are in the JSON.

| Pipeline | Rate | Items / questions | Rate % | Original Wilson CI | Pooled question CI | Source-stratified question CI |
|---|---|---:|---:|---|---|---|
| qwen2.5:3b / bin_ref | false_accept_rate | 173 / 118 | 1.73 | [0.59, 4.97] | [0.00, 3.95] | [0.00, 3.91] |
| qwen2.5:3b / bin_ref | false_reject_rate | 227 / 136 | 31.28 | [25.60, 37.58] | [24.17, 38.67] | [25.58, 36.77] |
| llama3.2:3b / bin_ref | false_accept_rate | 173 / 118 | 27.17 | [21.09, 34.24] | [20.11, 34.88] | [20.44, 34.30] |
| llama3.2:3b / bin_ref | false_reject_rate | 227 / 136 | 34.36 | [28.49, 40.75] | [25.74, 43.66] | [30.52, 38.39] |
| phi4-mini:3.8b / bin_ref | false_accept_rate | 173 / 118 | 7.51 | [4.44, 12.43] | [3.01, 12.65] | [3.09, 12.77] |
| phi4-mini:3.8b / bin_ref | false_reject_rate | 227 / 136 | 21.59 | [16.73, 27.39] | [14.83, 29.09] | [16.53, 26.73] |
| gemma3:4b / bin_ref | false_accept_rate | 173 / 118 | 15.03 | [10.47, 21.11] | [9.03, 21.77] | [9.26, 21.55] |
| gemma3:4b / bin_ref | false_reject_rate | 227 / 136 | 5.29 | [3.05, 9.01] | [2.45, 8.64] | [2.61, 8.37] |
| qwen3.5:4b / bin_ref | false_accept_rate | 173 / 118 | 13.29 | [9.02, 19.16] | [7.56, 19.41] | [7.83, 18.95] |
| qwen3.5:4b / bin_ref | false_reject_rate | 227 / 136 | 1.76 | [0.69, 4.44] | [0.41, 3.70] | [0.42, 3.57] |
| qwen2.5:3b / strict_ref | false_accept_rate | 173 / 118 | 1.16 | [0.32, 4.12] | [0.00, 2.99] | [0.00, 2.91] |
| qwen2.5:3b / strict_ref | false_reject_rate | 227 / 136 | 33.92 | [28.08, 40.30] | [26.94, 40.93] | [28.50, 39.17] |
| llama3.2:3b / strict_ref | false_accept_rate | 173 / 118 | 8.09 | [4.88, 13.12] | [3.98, 12.97] | [3.98, 12.79] |
| llama3.2:3b / strict_ref | false_reject_rate | 227 / 136 | 33.04 | [27.25, 39.40] | [24.78, 41.94] | [29.22, 37.10] |
| phi4-mini:3.8b / strict_ref | false_accept_rate | 173 / 118 | 7.51 | [4.44, 12.43] | [2.99, 12.71] | [3.18, 12.78] |
| phi4-mini:3.8b / strict_ref | false_reject_rate | 227 / 136 | 24.67 | [19.51, 30.67] | [17.24, 32.64] | [19.64, 29.73] |
| gemma3:4b / strict_ref | false_accept_rate | 173 / 118 | 13.29 | [9.02, 19.16] | [7.83, 19.66] | [7.91, 19.08] |
| gemma3:4b / strict_ref | false_reject_rate | 227 / 136 | 2.64 | [1.22, 5.65] | [0.45, 5.26] | [0.47, 5.22] |
| qwen3.5:4b / strict_ref | false_accept_rate | 173 / 118 | 8.67 | [5.32, 13.81] | [3.80, 14.04] | [4.09, 13.73] |
| qwen3.5:4b / strict_ref | false_reject_rate | 227 / 136 | 7.05 | [4.38, 11.14] | [3.31, 11.60] | [3.54, 11.01] |
| qwen2.5:3b / score_ref | false_accept_rate | 173 / 118 | 0.58 | [0.10, 3.20] | [0.00, 1.82] | [0.00, 1.82] |
| qwen2.5:3b / score_ref | false_reject_rate | 227 / 136 | 60.79 | [54.31, 66.91] | [53.95, 67.70] | [54.74, 66.67] |
| llama3.2:3b / score_ref | false_accept_rate | 173 / 118 | 43.35 | [36.19, 50.80] | [35.12, 51.04] | [36.14, 50.30] |
| llama3.2:3b / score_ref | false_reject_rate | 227 / 136 | 21.59 | [16.73, 27.39] | [14.47, 29.17] | [16.28, 27.06] |
| phi4-mini:3.8b / score_ref | false_accept_rate | 173 / 118 | 32.37 | [25.85, 39.66] | [24.24, 40.44] | [25.00, 39.89] |
| phi4-mini:3.8b / score_ref | false_reject_rate | 227 / 136 | 3.08 | [1.50, 6.23] | [1.00, 5.42] | [1.23, 5.39] |
| gemma3:4b / score_ref | false_accept_rate | 173 / 118 | 41.62 | [34.53, 49.07] | [34.59, 49.15] | [34.59, 48.75] |
| gemma3:4b / score_ref | false_reject_rate | 227 / 136 | 0.44 | [0.08, 2.45] | [0.00, 1.42] | [0.00, 1.38] |
| qwen3.5:4b / score_ref | false_accept_rate | 173 / 118 | 45.66 | [38.42, 53.10] | [37.89, 53.57] | [37.80, 53.45] |
| qwen3.5:4b / score_ref | false_reject_rate | 227 / 136 | 0.88 | [0.24, 3.15] | [0.00, 2.24] | [0.00, 2.25] |
| qwen2.5:3b / bin_free | false_accept_rate | 173 / 118 | 42.77 | [35.64, 50.23] | [34.76, 50.85] | [35.50, 50.28] |
| qwen2.5:3b / bin_free | false_reject_rate | 227 / 136 | 54.63 | [48.13, 60.97] | [47.11, 62.19] | [47.39, 62.10] |
| llama3.2:3b / bin_free | false_accept_rate | 173 / 118 | 42.77 | [35.64, 50.23] | [34.28, 51.32] | [36.53, 48.88] |
| llama3.2:3b / bin_free | false_reject_rate | 227 / 136 | 66.08 | [59.70, 71.92] | [57.89, 73.83] | [58.00, 73.83] |
| phi4-mini:3.8b / bin_free | false_accept_rate | 173 / 118 | 62.43 | [55.01, 69.30] | [54.82, 69.54] | [55.88, 68.85] |
| phi4-mini:3.8b / bin_free | false_reject_rate | 227 / 136 | 32.16 | [26.42, 38.49] | [25.10, 39.81] | [25.00, 40.47] |
| gemma3:4b / bin_free | false_accept_rate | 173 / 118 | 73.41 | [66.37, 79.43] | [66.46, 80.00] | [67.43, 79.21] |
| gemma3:4b / bin_free | false_reject_rate | 227 / 136 | 19.82 | [15.16, 25.49] | [14.03, 25.98] | [14.09, 25.99] |
| qwen3.5:4b / bin_free | false_accept_rate | 173 / 118 | 86.13 | [80.19, 90.50] | [80.47, 91.20] | [80.98, 91.06] |
| qwen3.5:4b / bin_free | false_reject_rate | 227 / 136 | 4.85 | [2.73, 8.47] | [2.15, 8.00] | [2.16, 8.10] |
| J5-majority / bin_ref | false_accept_rate | 173 / 118 | 7.51 | [4.44, 12.43] | [3.45, 12.14] | [3.49, 12.16] |
| J5-majority / bin_ref | false_reject_rate | 227 / 136 | 17.62 | [13.22, 23.10] | [11.85, 24.22] | [13.06, 22.13] |
| J5-unanimity / bin_ref | false_accept_rate | 173 / 118 | 0.58 | [0.10, 3.20] | [0.00, 1.86] | [0.00, 1.83] |
| J5-unanimity / bin_ref | false_reject_rate | 227 / 136 | 43.61 | [37.32, 50.12] | [35.45, 51.77] | [39.17, 48.23] |
| J3-majority / bin_ref | false_accept_rate | 173 / 118 | 9.83 | [6.23, 15.17] | [5.08, 15.08] | [5.20, 14.97] |
| J3-majority / bin_ref | false_reject_rate | 227 / 136 | 5.73 | [3.38, 9.55] | [2.84, 9.13] | [3.03, 8.86] |
| J3-unanimity / bin_ref | false_accept_rate | 173 / 118 | 3.47 | [1.60, 7.36] | [0.56, 7.25] | [0.57, 7.41] |
| J3-unanimity / bin_ref | false_reject_rate | 227 / 136 | 22.91 | [17.92, 28.80] | [15.92, 30.36] | [18.06, 27.94] |
| J5-majority / strict_ref | false_accept_rate | 173 / 118 | 4.62 | [2.36, 8.86] | [1.18, 8.82] | [1.18, 8.82] |
| J5-majority / strict_ref | false_reject_rate | 227 / 136 | 18.94 | [14.38, 24.54] | [12.90, 25.78] | [14.29, 23.68] |
| J5-unanimity / strict_ref | false_accept_rate | 173 / 118 | 0.00 | [0.00, 2.17] | [0.00, 0.00] | [0.00, 0.00] |
| J5-unanimity / strict_ref | false_reject_rate | 227 / 136 | 44.49 | [38.17, 51.00] | [36.41, 52.44] | [40.08, 48.92] |
| J3-majority / strict_ref | false_accept_rate | 173 / 118 | 8.09 | [4.88, 13.12] | [3.64, 13.14] | [3.66, 13.10] |
| J3-majority / strict_ref | false_reject_rate | 227 / 136 | 7.93 | [5.07, 12.18] | [3.96, 12.61] | [4.11, 12.23] |
| J3-unanimity / strict_ref | false_accept_rate | 173 / 118 | 2.89 | [1.24, 6.59] | [0.56, 6.01] | [0.56, 5.99] |
| J3-unanimity / strict_ref | false_reject_rate | 227 / 136 | 26.43 | [21.12, 32.53] | [18.86, 34.45] | [21.43, 31.45] |
| J5-majority / bin_free | false_accept_rate | 173 / 118 | 65.32 | [57.96, 72.01] | [57.86, 72.00] | [59.19, 71.35] |
| J5-majority / bin_free | false_reject_rate | 227 / 136 | 30.84 | [25.19, 37.12] | [23.93, 38.26] | [23.83, 38.14] |
| J5-unanimity / bin_free | false_accept_rate | 173 / 118 | 23.12 | [17.46, 29.95] | [15.72, 30.64] | [17.92, 28.48] |
| J5-unanimity / bin_free | false_reject_rate | 227 / 136 | 78.85 | [73.09, 83.66] | [72.08, 84.91] | [72.56, 84.96] |
| J3-majority / bin_free | false_accept_rate | 173 / 118 | 78.03 | [71.29, 83.56] | [71.58, 84.02] | [72.46, 83.43] |
| J3-majority / bin_free | false_reject_rate | 227 / 136 | 14.98 | [10.92, 20.20] | [9.79, 20.76] | [9.58, 20.66] |
| J3-unanimity / bin_free | false_accept_rate | 173 / 118 | 53.18 | [45.76, 60.46] | [44.74, 61.08] | [46.15, 60.00] |
| J3-unanimity / bin_free | false_reject_rate | 227 / 136 | 40.53 | [34.35, 47.02] | [32.66, 48.65] | [32.76, 48.87] |
| J5-mean_score / score_ref | false_accept_rate | 173 / 118 | 7.51 | [4.44, 12.43] | [3.68, 12.07] | [3.61, 12.12] |
| J5-mean_score / score_ref | false_reject_rate | 227 / 136 | 26.87 | [21.53, 32.99] | [19.81, 34.50] | [22.31, 31.42] |
| J3-mean_score / score_ref | false_accept_rate | 173 / 118 | 27.75 | [21.61, 34.85] | [20.71, 34.88] | [21.43, 34.46] |
| J3-mean_score / score_ref | false_reject_rate | 227 / 136 | 0.88 | [0.24, 3.15] | [0.00, 2.27] | [0.00, 2.26] |

## Reproduce

```bash
python scripts/e1_cluster_sensitivity.py
python scripts/e1_cluster_sensitivity.py --check
```

The JSON records input and implementation SHA-256 hashes, numpy/PRNG versions, cluster sizes and
all numerical results. Before reporting it checks the frozen sample, complete judgment matrix,
and agreement with all 34 original pipeline metrics and the original H1/family A intervals.
CI and tests independently regenerate both new artifacts. Canonical results are written with
ten significant digits and LF newlines, as in the existing analysis.
