Trey — I read your repo docs and the researcher guide, and I’ve pulled current provider details for GPT-5 thinking and Claude Sonnet 4.5. Here’s a concrete plan that balances power with cost, using your shared-control cache and batch/caching discounts.

# What we’re optimizing around

Your pipeline already does the right things: it treats each question as a paired comparison (control vs. treatment), computes McNemar/Wilcoxon with bootstrap CIs, and can reuse a single control run across many treatment runs via a shared-control cache. That last feature is key to keeping the bill down when you sweep nine system prompts.

Two recent facts guide cost choices:

- **OpenAI GPT-5 (thinking via the Responses API):** list price ≈ **$1.25/MTok input** and **$10/MTok output**; _Batch API is 50% off_; prompt caching applies automatically for repeated prefixes (“cached input” ≈ **$0.125/MTok**). Reasoning effort controls how many hidden “reasoning tokens” are produced; those are billed as **output tokens**. ([OpenAI][1])
- **Anthropic Claude Sonnet 4.5 (thinking/extended thinking):** **$3/MTok input** and **$15/MTok output** with prompt-cache pricing (**$3.75/MTok write**, **$0.30/MTok read**) and an advertised _up-to-50%_ saving with batch processing. Extended thinking requires a **minimum 1,024 thinking tokens**, which are billed as **output tokens** and count toward `max_tokens`. ([docs.claude.com][2])

# Design to run exactly twice (Anthropic + OpenAI), test all 9 prompts, and stay cheap

**Datasets.** Keep your existing split: (i) closed-book from TriviaQA / NQ-Open, (ii) open-book from SQuAD 2.0. Use one fixed item list (same items for **all 9 treatments + control** and for **both providers**) to preserve pairing and comparability. Your scorer already supports EM (closed/open), F1 (open), abstention, unsupported claims, and false-answer rates.

**Primary endpoints.**

- Closed-book: **EM** (primary), abstention as calibration secondary.
- Open-book: **Unsupported-claim rate** (primary; larger effects and better power than raw EM), F1 secondary. Your scorer exports both and your stats module handles McNemar, bootstrap CIs, FDR.

**Sample sizes that pass peer review without blowing budget.**
For a paired McNemar test, the minimum detectable EM delta obeys roughly
[
\delta_{\min}\approx (z_{1-\alpha/2}+z_{1-\beta})\sqrt{\frac{D}{N}},
]
where (D) is the discordant rate (P(\text{one correct only})). Using (\alpha=0.05) and 90% power gives (z\approx 3.24). With realistic (D=0.08)–0.12 for QA, **(N=2,500)** items yields (\delta\_{\min}\approx 1.7)–**2.0 pp**; **(N=5,000)** drives it near **1.1–1.6 pp**. (References for McNemar power/sample-size: MedCalc, Ristl, PASS/Stata.) ([MedCalc][3])

**Recommendation (per provider):**

- **Closed-book:** **N = 2,500** items. This comfortably detects a ~2 pp EM uplift and much larger uplifts if your excellence-style prompts replicate prior 6 pp effects at T=0.
- **Open-book:** **N = 2,500** items, powered on **unsupported-claim rate** (expect ~3–5 pp gains there; EM deltas will be smaller).

That’s **5,000 paired items** per provider. Because you reuse a single control run for all nine treatments, you submit **10 calls per item** (9 treatments + 1 control), not 19. Your shared-control cache takes care of the deduplication and downstream hydration.

**Reasoning/token budgets (cost-aware):**

- **OpenAI GPT-5 thinking (Responses API):** `reasoning.effort: minimal` for open-book and `reasoning.effort: medium` for closed-book; cap with `max_output_tokens` **768**. Reasoning tokens are billed as output; Batch API halves the price; prompt caching reduces your 1,391-token treatment system prompt cost by ~90% (OpenAI “cached input”). ([OpenAI Platform][4])
- **Claude Sonnet 4.5 (extended thinking):** set `thinking: {budget_tokens: 1024}` (the minimum) and `max_tokens: 1280` (closed-book) / **1536** (open-book) to leave room for short answers/quotes. Thinking tokens bill at output rate; use **Message Batches API** and **prompt caching** (mark the long system prompt as cacheable so it’s read at **$0.30/MTok** after a one-time write). ([Claude Docs][5])

**Multiple comparisons.** Pre-register nine treatment v. control hypotheses per provider × {closed, open}. Control FDR across those **18** tests (BH or Holm) — your stats runner already supports FDR. Report per-dataset McNemar and the meta-analysis you already compute.

# What this costs (with batch + caching)

Using conservative token caps and your long treatment prompts (≈1,391 tokens) with caching, here are realistic all-in estimates **per provider** for **N=2,500 closed + 2,500 open** (50,000 API calls/provider):

- **OpenAI GPT-5 thinking** (Batch 50% off; cached input on the long system prompt): **≈ $184** total (≈$169 output + $15 input). ([OpenAI Platform][6])
- **Anthropic Claude Sonnet 4.5** (Batches + prompt caching; 1,024 thinking tokens min): **≈ $524** total. Output dominates at $15/MTok. ([Claude][7])

**Grand total for both runs (both providers): ≈ $709.**
Even if you scale to **N=5,000 closed + 5,000 open** for ironclad power on sub-pp deltas, the combined estimate is ≈ $1.4k with the same settings. (Arithmetic shown on request; assumptions match the token budgets above.)

# How to run it with your repo (exact knobs)

1. **Freeze prompts & config.** Put the 9 treatments + control under `config/prompts/` (keep mirrors in `config/prompts copy/`), and define a `prompt_sets` entry with all ten prompts so the orchestrator expands trials deterministically.
2. **Build once, reuse everywhere.**

   - `python -m scripts.prepare_data --config config/eval_config.yaml` (write the fixed 2,500/2,500 lists).
   - `python -m scripts.build_batches --config config/eval_config.yaml --prompt_set default --temps 1.0` (reasoning runs).
     The run orchestrator will export the **shared control** to `experiments/run_<RID>/shared_controls/...` and hydrate it for all nine treatments.

3. **Anthropic run (first):** point to `claude-sonnet-4-5`, add:

   - `thinking: { budget_tokens: 1024 }` and `max_tokens` as above,
   - **Message Batches API** enabled,
   - prompt caching enabled/annotated for the system prompt.
     Submit via `scripts/run_all.py` with `--use_batch_api --max_concurrent_jobs` sized to your rate limits.

4. **OpenAI run (second):** switch provider/model to `gpt-5` via the Responses API, set `reasoning.effort` and `max_output_tokens=768`, and enable **Batch API**. OpenAI’s prompt caching is automatic for long, identical prefixes (≥1024 tokens), so your 1,391-token treatment system prompt gets the 90% “cached input” rate on repeats. ([OpenAI Platform][6])
5. **Post-processing:** `make parse | score | stats | report` — this writes `per_item_scores.csv`, `significance.json`, and a consolidated report; your stats module already supports McNemar, bootstrap CIs, meta-analysis, and FDR.

# Pre-registration notes to keep reviewers happy

- **Primary endpoints:** Closed-book EM and Open-book Unsupported-claim rate; two-sided α=0.05 with BH-FDR across 9 prompts per provider per task.
- **Power target:** 90% to detect **≈2 pp** improvements (McNemar paired) at (N=2{,}500) per task; justify with standard McNemar power formula and cite public calculators. ([homepage.univie.ac.at][8])
- **Sampling:** fixed random seed; stratify closed-book items across categories; keep the same item keys for both providers to preserve pairing.
- **Costs:** declare batch and caching usage up front (OpenAI: Batch 50% + cached input; Anthropic: Message Batches + prompt caching). ([OpenAI Platform][6])
- **Multiple comparisons:** report both per-prompt effects and an across-prompt meta-estimate (your code already computes fixed/random-effects with heterogeneity).

# Why this structure works

You’re exploiting all the big levers at once: (1) **paired design + shared control** for maximal statistical efficiency; (2) **batching** for 50% off on both providers; (3) **prompt caching** to neutralize the 1,391-token treatment system-prompt overhead; and (4) **tight reasoning budgets** so output tokens (the expensive part) don’t run away. That combination gets you strong, publication-grade statistics at a few hundred dollars per provider, while preserving apples-to-apples comparisons across nine treatments and two models.

If you want me to, I can drop in an example `eval_config.*.yaml` pair with the exact provider knobs (batch, caching annotations, thinking budgets, prompt set listing) and the one-liner run commands for both passes.

[1]: https://openai.com/api/pricing/?utm_source=chatgpt.com "API Pricing"
[2]: https://docs.claude.com/en/docs/about-claude/pricing?utm_source=chatgpt.com "Pricing"
[3]: https://www.medcalc.org/en/manual/sample-size-mcnemar-test.php?utm_source=chatgpt.com "Sample size calculation for McNemar test"
[4]: https://platform.openai.com/docs/guides/latest-model?utm_source=chatgpt.com "Using GPT-5 - OpenAI API"
[5]: https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/extended-thinking-tips?utm_source=chatgpt.com "Extended thinking tips"
[6]: https://platform.openai.com/docs/guides/batch?utm_source=chatgpt.com "Batch API"
[7]: https://www.claude.com/pricing?utm_source=chatgpt.com "Pricing"
[8]: https://homepage.univie.ac.at/robin.ristl/samplesize.php?test=mcnemar&utm_source=chatgpt.com "Sample size for McNemar's test of paired proportions"
