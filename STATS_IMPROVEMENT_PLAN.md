# Statistical Analysis Improvement Plan

This document describes why and how to strengthen the statistical analysis in this repository to robustly evaluate whether a treatment system prompt grounded in philosophy and ethics measurably reduces hallucinations compared to a control prompt. It is designed to be executable by another engineer/agent without prior context of this conversation.


## Why Improve the Analysis

- Scientific rigor: Current analysis reports averages and a few p-values but lacks effect sizes, confidence intervals for differences, multiple-comparisons control, and robust handling of small-sample/degenerate cases. A more principled approach is required to withstand peer review.
- Practical decision-making: We need clear uncertainty quantification and trade-off curves (e.g., abstention vs. correctness, cost per improvement) to guide product and research choices.
- Hallucination focus: We should center inference on hallucination-specific outcomes (false answers on unanswerables; unsupported claims) rather than only task accuracy.
- Reproducibility: Define primary/secondary endpoints, pre-specify procedures, and standardize outputs so analyses are repeatable and auditable across runs/trials/models.


## Current Pipeline (Summary)

Key files:
- `fireworks/parse_results.py`: Parses batch responses to `results/predictions.csv`.
- `scoring/score_predictions.py`: Computes per-item metrics (EM; F1 for open-book; abstain_rate; false_answer_rate for unanswerables; a simple `unsupported_rate` heuristic) and writes `results/per_item_scores.csv`.
- `scoring/stats.py`: Aggregates per-item scores per temperature and runs: McNemar test on binarized outcomes and Wilcoxon on deltas; writes `results/significance.json`.
- `scripts/generate_report.py`: Produces `reports/report.md` with per-temp tables, simple 95% CIs for single-condition means (not differences), and significance summaries.
- `scripts/compare_runs.py`: Prints descriptive comparisons between two experiment runs.

Limitations:
- Binarization and averaging of replicates before testing loses information; no effect sizes or CIs for differences; no multiple-comparisons control; simplistic unsupported heuristic; no subgroup analysis, selective-risk curves, or mixed-effects modeling; limited robustness for small n.


## Goals and Principles

- Pre-specify endpoints: define primary and secondary outcomes to reduce multiplicity/p-hacking.
- Use paired, replicate-aware inference whenever available; otherwise pair by item.
- Report effect sizes with CIs alongside p-values and adjust for multiple comparisons where applicable.
- Focus hallucination metrics: prioritize false answers on unanswerables and evidence support for claims.
- Provide robustness checks: permutation tests, bootstrap CIs, sensitivity analyses, subgroup summaries.
- Keep outputs machine-readable and documented so reports and downstream tools can rely on stable schemas.


## Endpoints and Definitions

- Conditions: `control` vs `treatment` prompts.
- Types: `closed` (no context) and `open` (with context/unanswerables).
- Metrics (per item, averaged over replicates unless otherwise stated):
  - Accuracy: EM (both types), F1 (open only).
  - Abstention: `abstain_rate` (fraction abstained); Coverage = 1 − abstain_rate.
  - Hallucination proxies (open only):
    - `false_answer_rate`: answering when question is unanswerable.
    - `unsupported_rate`: predicted text not supported by context (to be improved).
  - Selective-risk: selective accuracy = accuracy conditional on answering.

Primary outcome (recommendation):
- Open-book unanswerables: difference in `false_answer_rate` (treatment − control). Lower is better.

Secondary outcomes:
- Accuracy (EM/F1), `abstain_rate`, `unsupported_rate`, selective accuracy, and coverage.


## Deliverables and Artifacts (Required)

- Extended `results/significance.json` including per-temp statistics with effect sizes, CIs, and adjusted q-values (schema below).
- Updated `reports/report.md` showing Δ (difference) with 95% CI, exact McNemar (with OR and CI), and FDR-corrected q-values.
- Optional new artifacts (later phases): subgroup summaries, risk–coverage curves, power analysis outputs.


## Implementation Plan (Phased)

### Phase 1 — Core Robust Inference (Must Do)

1) Exact McNemar for paired binary outcomes
- For each temperature and item, construct discordant counts between control vs treatment on a binary outcome (e.g., EM ≥ 0.5 as correct/incorrect).
- Use exact binomial test on discordant pairs (b = 0→1, c = 1→0). Report:
  - p_exact
  - Odds ratio (OR) with 95% exact CI (Baptista–Pike) or mid-p variant for small counts.
- Prefer replicate-level pairing when replicate indices align across conditions; otherwise pair by item.

2) Paired bootstrap CIs for differences (Δ)
- For metrics in {EM, F1, abstain_rate, false_answer_rate, unsupported_rate}:
  - Compute per-item difference Δ_i = metric_treatment_i − metric_control_i.
  - Bootstrap over items (B ≥ 5,000; configurable): sample items with replacement, compute mean(Δ*) per replicate, and derive percentile (or BCa) 95% CI. Report mean Δ and CI.
  - Also compute Hodges–Lehmann estimator (median of all pairwise averages or median of Δ_i) and Wilcoxon signed-rank p-value with effect size r.
  - Effect sizes: paired Cohen’s d on Δ (mean(Δ)/sd(Δ)) and Cliff’s delta on Δ for distribution-free effect.
- Robustness guards: handle n < 2 (return NaNs with notes), all-zero-Δ cases (return Δ=0, p=1).

3) Permutation test (paired label-flip)
- For ΔEM and ΔF1: conduct N_perm (e.g., 5,000) sign-flip permutations on per-item deltas to obtain non-parametric p-values. Include when feasible (toggle in config if runtime is a concern).

4) Extended JSON schema for `results/significance.json`
- Per temperature key (stringified float):
```json
{
  "0.7": {
    "mcnemar": {
      "b": 15,
      "c": 5,
      "p_exact": 0.0123,
      "odds_ratio": 3.00,
      "or_ci": [1.25, 7.85]
    },
    "paired": {
      "em": {
        "mean_delta": 0.021,
        "ci": [0.008, 0.035],
        "p_wilcoxon": 0.018,
        "hl_estimate": 0.020,
        "cohens_d": 0.18,
        "cliffs_delta": 0.12,
        "p_permutation": 0.022
      },
      "f1": { ... },
      "abstain_rate": { ... },
      "false_answer_rate": { ... },
      "unsupported_rate": { ... }
    },
    "fdr": {
      "pvals": { "em": 0.018, "f1": 0.044, ... },
      "qvals": { "em": 0.036, "f1": 0.066, ... }
    }
  }
}
```
- Note: if not all metrics apply to a type (e.g., F1 for closed-book), include null or omit.

5) Reporting updates (`scripts/generate_report.py`)
- Add per-temp Delta section: for each metric, show mean Δ with 95% CI, Wilcoxon p, Cohen’s d, Cliff’s delta, and permutation p if available.
- Show exact McNemar b/c, p_exact, OR with CI.
- Include FDR-adjusted q-values for all metric tests within that temperature (see Phase 2).

6) CLI integration
- Ensure `make stats` or `make report` runs the new stats pipeline. Maintain deterministic seeds for bootstrap/permutation via config.


### Phase 2 — Multiple Comparisons and Subgroups (High Priority)

1) Multiple comparisons control
- Across tests within a run (temps × metrics × types × subgroups), compute Benjamini–Hochberg FDR-adjusted q-values.
- Store per-temp `fdr.qvals` and echo them in the report.

2) Subgroup analyses
- Stratify by dataset (`triviaqa`, `nq_open`, `squad_v2`) and optional difficulty proxies (e.g., question length tertiles).
- Compute the same paired stats per subgroup. Provide a small forest table per temp with subgroup effects and CIs.


### Phase 3 — Abstention and Selective Risk (High Value)

1) Selective risk metrics
- For each item: compute coverage = 1 − abstain_rate and selective accuracy = accuracy among non-abstentions.
- Plot/compute risk–coverage curves per condition and summarize area under risk–coverage (AURC). If plotting is out of scope, provide discrete points at defined abstention thresholds.

2) Non-inferiority tests (TOST)
- Pre-specify a non-inferiority margin for EM/F1 and test whether treatment is not worse than control beyond the margin while improving hallucination metrics.


### Phase 4 — Improved Unsupported Detection (Sensitivity)

1) New `scoring/unsupported.py`
- Implement multiple strategies:
  - Baseline: normalized-substring (current).
  - Overlap-based: token overlap and sentence-level alignment between prediction and provided context with conservative thresholding.
  - Optional NLI: entailment score between context (premise) and claim (hypothesis) using a local model. Threshold decisions to mark support vs unsupported.

2) Sensitivity analysis
- Sweep `unsupported_threshold` and produce treatment effect across thresholds (plot or tabulate) to show robustness.


### Phase 5 — Mixed-Effects Modeling (Optional, Strong)

1) Replicate-level dataset construction
- Build a long-format table with columns: item_id, dataset, type, temp, condition, replicate_index, binary_correct (EM≥0.5), F1 (if open), abstained, unanswerable, etc.

2) Logistic mixed model
- Outcome: binary_correct. Fixed effects: condition, temp, condition×temp. Random intercepts: item (and optionally dataset). Report condition OR with 95% CI and robust SE.

3) Continuous outcome model for F1
- Linear mixed model on F1 or beta regression on (F1′ in (0,1)). Report treatment effect with 95% CI.

4) Integrate summaries into report as a robustness check.


### Phase 6 — Power and Minimum Detectable Effect (MDE)

- New `scripts/power_analysis.py`:
  - Given item count and observed variance or baseline rates, compute MDE for primary endpoint at α=0.05 (two-sided) and for a chosen power (e.g., 80%).
  - Provide guidance on required N for planned improvements.


### Phase 7 — Cost-Effectiveness

- Extend `scripts/summarize_costs.py` or add a small utility to compute:
  - Cost per 1-point (absolute) reduction in `false_answer_rate` and per 1-point gain in EM/F1.
  - Summarize at each temperature to aid in selecting operating points.


## Code Changes (File-Level Checklist)

- `scoring/stats.py` (extend)
  - Load per-item scores; build paired arrays per temperature (and per subgroup in Phase 2).
  - Implement:
    - Exact McNemar (p_exact, OR, CI) on binary outcomes (e.g., EM≥0.5).
    - Paired bootstrap CIs for Δ across metrics; Wilcoxon p; Hodges–Lehmann; Cohen’s d; Cliff’s delta; optional permutation p.
    - FDR adjustment across all tests at a given run (Phase 2).
  - Write extended `significance.json` with schema above.

- `scripts/generate_report.py` (extend)
  - Parse extended `significance.json`.
  - Render Δ with 95% CIs and effect sizes; render exact McNemar with OR and CI; include q-values.
  - Add optional subgroup sections and risk–coverage summaries.

- `scripts/compare_runs.py` (extend)
  - Display Δ with CIs and significance (p and q) per metric/temperature.

- `scoring/unsupported.py` (new; Phase 4)
  - Provide support-check functions and sensitivity sweep utilities.

- `scripts/power_analysis.py` (new; Phase 6)
  - CLI to compute MDE/power and emit JSON/markdown.


## Algorithms (Concise Pseudocode)

Exact McNemar (paired binary):
```
for temp in temps:
  b = count(items where ctrl=0 and trt=1)
  c = count(items where ctrl=1 and trt=0)
  p_exact = 2 * min(BinomialCDF(min(b,c), b+c, 0.5), 1 - BinomialCDF(min(b,c)-1, b+c, 0.5))
  or_est = (b+0.5)/(c+0.5)  # Baptista–Pike continuity
  or_ci = exact_CI_baptista_pike(b, c, alpha=0.05)
```

Paired bootstrap for Δ:
```
Δ = [trt_i - ctrl_i for i in items]
for r in 1..B:
  sample = resample_with_replacement(Δ)
  boot_means[r] = mean(sample)
ci = percentile(boot_means, [2.5, 97.5])  # or BCa
mean_delta = mean(Δ)
cohens_d = mean_delta / std(Δ)
cliffs_delta = cliffs_delta_from_pairs(Δ)
```

Permutation (paired sign-flip):
```
Δ = [trt_i - ctrl_i]
for r in 1..N_perm:
  signs = random choice {+1,-1} per i
  perm_means[r] = mean(signs * Δ)
p_perm = fraction(|perm_means| >= |mean(Δ)|)
```

Benjamini–Hochberg FDR:
```
# Given p-values p1..pm for a family of tests
sort p ascending with indices
q_k = min_{j>=k}( m/j * p_(j) )
```


## Configuration and Determinism

- Add to `config/eval_config.yaml` (examples):
```
stats:
  bootstrap_samples: 5000
  permutation_samples: 5000
  random_seed: 1337
  enable_permutation: true
  enable_fdr: true
```
- Use `random_seed` to seed `numpy.random.default_rng(seed)` consistently.


## Testing and Validation

- Smoke tests: `make smoke` should complete and generate `significance.json` with extended fields and a report with Δ + CI sections.
- Degeneracy tests: small subsets where b+c=0, all Δ=0, or n=1 should run without crashing and produce sensible neutral outputs.
- Consistency: repeated runs with fixed seed produce identical `significance.json`.
- Optional unit tests: add under `tests/` for bootstrap CI and McNemar exact p for small constructed examples.


## Acceptance Criteria

- `results/significance.json` contains per-temp exact McNemar stats, paired bootstrap CIs and effect sizes for Δ across applicable metrics, and (when enabled) FDR-adjusted q-values.
- `reports/report.md` includes:
  - For each temperature/type: tables of means plus a Delta section with Δ and 95% CI; effect sizes; exact McNemar (b, c, p_exact, OR, CI); and q-values.
  - If subgroup analysis enabled: per-dataset effect tables.
- Scripts run via `make eval` produce the above without errors on smoke data.


## Risks and Mitigations

- Small n / ties / all-zero deltas: guard paths return neutral values (p=1, Δ=0, CI=[0,0]).
- Replicate misalignment: if `sample_index` mismatch across conditions, fall back to item-level pairing only.
- Multiple comparisons inflation: BH-FDR reduces false discoveries; pre-specify a single primary endpoint.
- Unsupported detection errors: provide baseline and advanced variants; add sensitivity analysis across thresholds.


## How to Run (after implementation)

- Standard pipeline: `make eval` or `python -m scripts.run_all --config config/eval_config.yaml --archive`
- Rebuild only stats/report on existing results:
  - `python -m scoring.stats --per_item_csv results/per_item_scores.csv --metric em --out_path results/significance.json`
  - `python -m scripts.generate_report --config config/eval_config.yaml --results_dir results --reports_dir reports`


## Roadmap Summary

1) Phase 1 (core inference): exact McNemar, bootstrap CIs for Δ, effect sizes, permutation p, report integration.
2) Phase 2 (FDR + subgroups): BH q-values, dataset-level summaries.
3) Phase 3 (selective risk): coverage/accuracy curves and AURC; non-inferiority tests.
4) Phase 4 (unsupported upgrade): new module and sensitivity analysis.
5) Phase 5 (mixed-effects): replicate-level logistic/linear mixed models.
6) Phase 6 (power/MDE): planning utilities.
7) Phase 7 (cost-effectiveness): cost-per-improvement metrics.


---
This plan intentionally separates statistical methodology (Phase 1) from modeling enhancements (Phases 4–5) to deliver immediate rigor while allowing incremental sophistication over time.
