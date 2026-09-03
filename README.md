# scaffold-sensitivity-protocolqa

Does measured benchmark score depend on the scaffold around a model, not just
the model itself? Two scaffolds (a plain single call vs. a decompose-into-
sub-questions-then-recombine chain) run against the same model
(`claude-haiku-4-5`) on the same 108-item benchmark
([LAB-Bench ProtocolQA](https://arxiv.org/abs/2407.10362), FutureHouse Inc.),
via Inspect AI (`inspect_evals`).

Full design, rules, and provenance: [`CLAUDE.md`](CLAUDE.md).
Aggregate results and methodology detail: [`public/report.txt`](public/report.txt).

## Result

Accuracy was unchanged. Single arm 0.528 and 0.556 across two replicates, chain
arm 0.491 and 0.528. All four sit inside the same noise band.

Refusals were not. The chain arm declined on 12 and 9 of 108 items versus 3 and 2
for the single arm, on items the single arm mostly answered normally. The
refusals were scattered across replicates rather than concentrated on a stable
subset, so the effect is aggregate rather than per-item.

Reproducibility looks worse under the chain scaffold. Within-arm agreement across
replicates was kappa 0.588 for single and 0.438 for chain, a gap of -0.150 with a
95% bootstrap interval of [-0.356, 0.057]. The point estimate stays negative when
refusals are excluded. The interval crosses zero, so this is suggestive and
underpowered at n=108 rather than established.

The four runs reported here are exploratory. A power analysis run afterward puts
a confirmatory study at 8 replicates per arm if the true gap matches the pilot
estimate, and 18 if the true gap is 0.10. That figure is recorded in CLAUDE.md as
a planning estimate. No confirmatory replicate count has been committed to and no
confirmatory runs have been made.

## Safety commitments

This project touches a biology-lab-protocol benchmark, so before anything
else:

1. **No novel hazardous content.** Every question, answer choice, and
   protocol excerpt involved comes from LAB-Bench ProtocolQA, a published
   academic benchmark that already exists in the literature. Nothing here
   generates new hazardous information, new protocols, or new dual-use
   content — the model is only ever asked to answer or decompose questions
   that were already public before this project started.
2. **Generic decomposition prompts.** The "chain" scaffold's decomposition
   step (`scaffold_study.py`) is a domain-agnostic instruction — "break this
   multiple-choice question into up to 3 self-contained sub-questions" — with
   nothing biology-specific or hazard-specific in it. The identical prompt
   would apply unchanged to a history or math benchmark; it isn't tuned to
   extract or elicit anything.
3. **Aggregate rates only, in public.** This repo publishes accuracy,
   agreement (Cohen's kappa), and refusal-rate statistics, plus per-item
   *outcome labels* (correct/wrong/refusal, as plain categorical flags) for
   independent verification of those statistics. It does **not** publish raw
   model completions, sub-question text, or answer-choice text anywhere in
   this repo — see "What's excluded" below.
4. **This measures score validity, not attack efficacy.** The question under
   study is methodological: does a multi-call scaffold change a model's
   measured score and reproducibility on an existing benchmark, and how
   would you know? This is not a jailbreak, elicitation, or red-teaming
   study, and nothing here is intended to demonstrate or improve any
   technique for extracting unsafe outputs from a model.

## What's in this repo

- `scaffold_study.py` — the two Inspect AI tasks (`single_arm`, `chain_arm`).
- `analyze_study.py` — agreement/kappa analysis over completed eval logs.
- `power_analysis.py` — replicate-count planning from pilot data (see
  `CLAUDE.md` → Provenance for why 8 replicates/arm was the resulting figure,
  and why a simpler, more conservative method was used over a fancier one
  that turned out to embed a bias).
- `prepare_public_release.py` — strips free-text columns out of the local
  analysis CSVs to produce everything under `public/`.
- `public/rows.csv`, `public/chain_refusal_detail.csv` — per-item outcome
  labels only (arm, replicate, sample id, correct/wrong/refusal, boolean
  flags, sub-question counts). No question text, no answer text, no model
  output.
- `public/report.txt` — the full aggregate report (accuracy, refusal rates,
  within- and between-arm kappa with bootstrap confidence intervals).

## What's excluded

- Raw Inspect eval logs (`logs/*.eval`) — these contain full model
  transcripts, including sub-question text and final answers, for every
  item. Kept local only.
- The unstripped local analysis outputs (`analysis/rows.csv`,
  `analysis/chain_refusal_detail.csv`) — these carry sub-question text,
  model completions, and answer-choice text. Kept local only;
  `prepare_public_release.py` is the one-way filter from these into
  `public/`.

## Reproducing this

See `CLAUDE.md` for the exact `inspect eval` commands, the pre-registered
replicate count for any confirmatory follow-up, and full provenance
(exploratory vs. confirmatory runs, the power analysis behind the replicate
count, and the caveats on both).
