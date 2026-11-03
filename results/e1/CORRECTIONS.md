# E1 corrections and run integrity (3 October 2026)

Moved here from the README on 2026-11-03; the text is unchanged apart from link paths.

- `python scripts/e1_audit_time_gold.py` rechecks all **107 deduplicated time candidates** and the frozen pilot,
  main, robustness and pairwise sets. [The correction audit](time_gold_audit.json) records ten
  `ambiguous → correct` changes: eight identifier errors and two date-year errors. The latter two answers are
  tentative ("typically" / "around"); exact clock matching does not settle their meaning. No selected item's label
  changes. The original pool, selection order, replies and first-run manifest are retained, and no missing judge
  responses are invented for the newly eligible answers. A changed sample would require a separate experiment.
- Subset cost reports now sum only responses belonging to the selected items. Previously, a 152-item source subset
  for a single judge showed **2.63 calls/item** because all 400 calls were charged to it; the correct value is **1**.
  `per_source.json` and `sensitivity_phrase_gold.json` have been regenerated. Full-set accuracy, ranking and
  aggregation results are unchanged.
- Analyses require the declared judge × condition × item matrix (both orders for pairs), reject duplicate and
  unexpected cells, and validate all main-analysis inputs before writing reports. A real reply with `parsed=null`
  remains a costed parse failure; an absent record stops analysis.
- The judge runner atomically checkpoints new records without removing other judges, checks item/prompt/options/
  model identities on resume, and takes a per-set writer lock. The original main manifest is immutable. Complete
  legacy slices are an offline no-op; incomplete legacy data must use a new run because its full provenance cannot
  be reconstructed. A lock left by a hard crash should be removed only after confirming its process has stopped.
