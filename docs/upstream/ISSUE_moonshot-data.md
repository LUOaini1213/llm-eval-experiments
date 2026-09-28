Posted as https://github.com/aiverify-foundation/moonshot-data/issues/222 (links filled in when posted).

**Title:** Answer-key errors and metadata typos in the Singapore MCQ datasets; `exactstrmatch` scores mostly answer format on the TF/MCQ recipes

---

While auditing the ten Apache-2.0 Singapore datasets in this repository (commit `30fac12`, 360 items), I found three multiple-choice answer keys that disagree with their own options. For each one I checked the fact against an official source (checked 2026-09-28):

| Item | Problem | Official source |
|---|---|---|
| `singapore-transport-system` #15 (0-based) | The key is `B) 1987`, but option B is `1990`. `1987` is option A. PTC was established in 1987. | [ptc.gov.sg – About PTC](https://www.ptc.gov.sg/who-we-are/about-ptc/): "Established in 1987 under the Public Transport Council Act" |
| `singapore-transport-system` #26 | The key is `B) 2001`, but option B is `2000` and no option is 2001, so no answer can be scored correct. | [sbstransit.com.sg – Milestones, 2001](https://www.sbstransit.com.sg/milestones): "Singapore Bus Services Limited changed its name to SBS Transit Ltd" |
| `singapore-public-housing` #2 | The key is `B) 5 years`, but option B is `7 years`. `5 years` is option A. The question also doesn't say which flat classification it means (MOP is 5 years for unclassified/Standard flats, 10 years for Plus/Prime). | [hdb.gov.sg – Eligibility for selling a flat](https://www.hdb.gov.sg/managing-my-home/selling-a-flat/eligibility) |

With a letter-based scorer, these keys fail a model that picks the right option.

**Other data issues (not answer-key errors):**
- `singapore-public-housing.json` has `"name": "Singapore Transport System"`.
- Name typos: "Singapore Polical History"; "Singapore POMFA …" in the three POFMA datasets. These are also in `datasets/cache.json`.
- `singapore-public-housing` has two exact duplicate pairs: #0/#1 and #9/#10.
- The first example of `sg-legal-glossary` is a CSV header row (`{"input": "Term", "target": "Explanation"}`).

**`exactstrmatch` on `singapore-facts-tf` / `singapore-facts-mcq`.** I ran both recipes against five small local models (Ollama). I then compared the stock `exactstrmatch` accuracy with a lenient parse of the same replies (the first TRUE/FALSE, or the leading option letter). The two often disagree by much more than any difference between models. For example, gemma3:4b scores 0% on all three TF datasets under `exactstrmatch`, while 70% of its replies give the keyed answer. The gap comes from answer format (markdown bold, a sentence around the answer, "B) 2001" vs "B"), not from knowledge.

Possible fixes, if useful:
- a normalising TF/MCQ metric (strip markup; take the first TRUE/FALSE or the option letter), or `exactstrmatch` with that normalisation as an option;
- a stricter output instruction in the `singapore-facts-tf` / `-mcq` prompt templates.

I've opened a small PR that fixes only the three keys and the name typos: [link]. I left the duplicates, the header row and `cache.json` out of it, because deleting items shifts indices, and I wasn't sure whether `cache.json` is generated.

Method, per-item flags and the metric comparison are in [link to github.com/LUOaini1213/llm-eval-experiments, E2 section]. This is an independent personal project, not affiliated with AI Verify Foundation.
