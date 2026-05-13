# LLM-as-Judge on a Budget

Variance-adaptive query allocation for LLM-as-a-judge evaluation under a fixed budget. Code accompanying the paper *LLM-as-Judge on a Budget* (Saha, Wagde, Kveton, 2026).

## What this project does

When using an LLM (e.g. GPT-4.1-nano, Llama-3.1-8B-Instruct) as a *judge* to score `K` prompt-response pairs, scores are stochastic: the same pair queried twice yields different numbers. To estimate the true mean score per pair, you query each pair multiple times. With a fixed query budget `B`, the question is **how to distribute `B` queries across the `K` pairs to minimize worst-case estimation error (WCE):**

```
WCE = max_{i in [K]} | s_i - s_hat_i |
```

Uniform allocation (`B/K` queries per pair) is provably suboptimal when per-pair score variance `σ²_i` is heterogeneous — easy pairs waste queries while hard pairs are under-sampled.

This project formalizes the allocation problem as a multi-armed bandit and implements two algorithms:

- **ROBIN** (known variance, oracle). At each step pull arm `argmax_i σ²_i / n_i(t)`. Provably achieves `WCE = Õ(sqrt(Σ σ²_i / B))`. Not implementable in practice (true variances unknown).
- **ROBIN-HOOD** (unknown variance, practical). Two phases:
  1. *Init-Exploration*: uniform pulls for `t_0 = 4 log(1/δ)` rounds per arm to seed variance estimates.
  2. *Adaptive*: pull arm `argmax_i V̄_i(t) / n_i(t)`, where `V̄_i(t)` is a UCB on the variance (Eq. 6 in paper).

  Achieves the same `Õ(sqrt(Σ σ²_i / B))` WCE with only logarithmic overhead from variance estimation.

Empirically (HelpSteer2, 1017 prompt-response pairs, 4 attributes, 2 judge LLMs) ROBIN-HOOD reaches the same WCE as Uniform with **~half the budget**.

## Repository contents

```
OptimalDesign.ipynb           — single notebook, runs all experiments + figures
new_data/                     — judge scores: 8 jsonl files
   helpsteer2_<attribute>_<model>.jsonl
   attribute ∈ {complexity, correctness, helpfulness, verbosity}
   model     ∈ {gpt-4.1-nano, llama-3-1-8b}
   each row: {"predicted_score": <float>, ...}; 30 scores per prompt-response pair, 1017 pairs ⇒ ~30k rows/file
outputs/exp1/                 — Experiment 1 plots (WCE vs budget per attribute)
outputs/exp2/                 — Experiment 2 tables (WCE at 50k & 100k budget, mean±std)
outputs/exp3/                 — Experiment 3 plots (correlation with human scores)
outputs/histograms/           — Experiment 4 plots (variance + mean histograms)
LLM-as-a-Judge-on-a-Budget.pdf — paper
```

The notebook alone is sufficient to reproduce all results.

## Data

> **Note:** The judge-score data in `new_data/` is **proprietary and not redistributed** in this repository. The schema and generation procedure are documented below so it can be regenerated. The exact prompts used to elicit scores from the judge LLMs are provided in the accompanying paper (`LLM-as-a-Judge-on-a-Budget.pdf`, appendix / experimental setup).

**Source.** 1017 prompt-response pairs sampled from HelpSteer2 (Wang et al. 2024) — an open dataset of prompt-response pairs with human ratings on multiple quality attributes.

**Judge LLMs.** Each prompt-response pair was scored by two judge models:
- `gpt-4.1-nano`
- `llama-3-1-8b` (Llama 3.1 8B Instruct)

**Attributes scored.** Each `(pair, judge)` combination was scored independently on four attributes: `complexity`, `correctness`, `helpfulness`, `verbosity`.

**Repetitions.** Each `(pair, judge, attribute)` triple was queried **30 times** (independent samples with non-zero temperature) to expose the stochasticity that the algorithms exploit. Total: 2 judges × 4 attributes × 1017 pairs × 30 reps ≈ **244k judge evaluations**.

**File layout.** `new_data/helpsteer2_<attribute>_<model>.jsonl` — one file per `(attribute, judge)` combination (8 files). Each file is JSON-lines; row `i` corresponds to the `(i // 30)`-th prompt-response pair, sample `i % 30`. Minimum required field:

```json
{"predicted_score": 3.0}
```

(Real files also carry the prompt, response, rationale, and judge metadata, but only `predicted_score` is consumed by the notebook.)

**Score range.** Floats in `[0, M]` (HelpSteer2 uses a Likert-like scale, typically 0–4). Invalid / unparseable judge outputs are stored as `-1` and filtered out at load time; prompt-response pairs that end up with zero valid scores are dropped (so the effective `K` is 1016 instead of 1017 in a few files).

**Regeneration.** To rebuild `new_data/` from scratch: take 1017 HelpSteer2 prompt-response pairs, then for each `(attribute, judge_model)` query the judge 30 times with the prompt template from the paper, parse the numeric score out of the rationale, and write to the corresponding `jsonl` file in the order above.

## Quick start

Requirements:

- Python ≥ 3.10
- `numpy`, `scipy`, `pandas`, `matplotlib`, `joblib`

Install:

```
pip install numpy scipy pandas matplotlib joblib
```

Run:

```
jupyter notebook OptimalDesign.ipynb
```

Execute cells top-to-bottom. Output PDFs land under `outputs/`.

## Notebook structure

| Section | Purpose |
|--|--|
| Cell 0 | Imports, matplotlib style |
| Main Graphs (cells 2–3) | **Experiment 1.** Loop over all 8 `(attribute, model)` files. For each, run Uniform / ROBIN / ROBIN-HOOD, average over `num_runs=50`, plot WCE vs budget at 10 budget checkpoints (`n/10 … n`). Saves PDF per file. |
| Experiment 2 Data Generation (cell 5) | **Experiment 2.** Same algorithms, but tabulates WCE (mean ± std) at exactly two budgets: 50k and 100k. Output CSV → `outputs/exp2/`. |
| Cell 6 | Formats Exp 2 CSV into `mean ± error_bar` table (`exp2 final 007.csv`). |
| Experiment 3 (cell 8) | **Correlation with human scores.** ROBIN-HOOD only. Compute Pearson `r`, Spearman `ρ`, Kendall `τ` between estimated mean scores and human (HelpSteer2) means at each budget checkpoint. |
| Cell 9 | Re-renders the small/main-paper version of the Exp 3 plot. |
| Experiment 4 (cells 11–13) | **Dataset statistics.** Histograms of per-pair score variance and per-pair mean score (Fig. 1, 2 of paper). |

## Algorithms (as coded)

The notebook uses three internal names: `Uniform`, `Variance`, `UCB` — which map to the paper as:

| Notebook name | Paper name | What it pulls |
|--|--|--|
| `Uniform` | Uniform | `argmax_i 1 / n_i(t)` (round-robin) |
| `Variance` | ROBIN  | `argmax_i σ²_i / n_i(t)` using true sample variance from all 30 scores |
| `UCB` | ROBIN-HOOD | Init phase: pull `arm = t mod K` until `floor(t/K) > 4 log(1/δ)`. Then `argmax_i V̄_i(t) / n_i(t)` where `V̄_i(t) = σ̂²_i / (1 − 2 sqrt(log(1/δ)/n_i(t)))`. |

(Note: in Experiment 1 (cell 3) the ROBIN-HOOD algorithm is labelled `ROBIN-HOOD` directly; in cell 5 it is labelled `UCB`. Same algorithm.)

## Key hyperparameters

| Symbol | Notebook var | Default | Meaning |
|--|--|--|--|
| `K` | `K` | 1017 | Number of prompt-response pairs (arms). Auto-computed from data. |
| `B` | `n` | 100000 | Total query budget per run. |
| `δ` | `delta` | 0.007 | Failure probability. Controls UCB tightness *and* warm-up length. |
| warm-up | derived | `4 K log(1/δ) ≈ 20164` | Uniform-exploration phase length for ROBIN-HOOD. |
| runs | `num_runs` | 50 | Independent runs to average over. |
| per-arm reward pool | `per_arm_rewards` | 30 | Pre-collected scores per pair; each "query" draws one uniformly at random. |

**Choosing δ.** Too small ⇒ very long warm-up, no room for adaptive gains within budget. Too large ⇒ UCB fails frequently, allocation drifts from optimal. `δ = 0.007` was tuned to work across all 8 `(attribute, model)` combinations. Cell 3 also includes `δ = 0.07` as a deliberately-poor example (paper Fig. 5).

## Reproducing paper figures

| Paper figure / table | Notebook cell | Output dir |
|--|--|--|
| Fig. 1, 2 (variance / mean histograms) | cells 11–13 | `outputs/histograms/` |
| Fig. 3, 4, 5 (WCE vs budget) | cell 3 | `outputs/exp1/` |
| Fig. 6, 7 (correlation with human scores) | cells 8–9 | `outputs/exp3/` |
| Tab. 2 (WCE at 50k / 100k) | cells 5–6 | `outputs/exp2/exp2 final 007.csv` |

## Runtime

50 runs × 100k steps × 1017 arms × 3 algorithms × 8 files ≈ several hours on a single machine. Reduce `num_runs` or `n` for quick exploration. The notebook is plain numpy — no GPU needed.

## Citation

```
@article{saha2026llmjudgebudget,
  title  = {LLM-as-Judge on a Budget},
  author = {Saha, Aadirupa and Wagde, Aniket and Kveton, Branislav},
  year   = {2026},
  note   = {arXiv:2602.15481}
}
```
