# scaffold-study

## The experiment

Same 108 questions, same model (`claude-haiku-4-5`), two setups. Question: does
the score depend on the scaffold rather than the model?

- **Arm "single"** — plain single call, just `generate()`.
- **Arm "chain"** — the model breaks the question into 2 to 3 sub-questions,
  answers each one in a FRESH context with no memory of the others, then reads
  those answers and picks a final answer to the original question.

Note: "fresh context" here means fresh *between sub-questions within one
item*, not between items. Every item is independent in both arms, as Inspect
does by default.

Each arm runs **twice** with independent sampling. Four runs total. The repeat
run is the noise floor — without it there's no way to tell whether
disagreement between arms means anything.

## Rules that must not be broken

1. Fresh context per sub-question. Sub-question 2 must not see the answer to
   sub-question 1. This is the whole point and the easiest thing to drop.
2. Maximum 3 sub-questions.
3. If decomposition fails or returns unparseable output, fall back to a
   single answer, but record it in `state.store` so it can be counted. A
   silent fallback would contaminate the chain arm with single-call results.
4. Refusals recorded separately from wrong answers.
5. Reuse the existing `inspect_evals` dataset and scorer. Do not rebuild them.
6. Same model in both arms.
7. Replicates need independent sampling. Not temperature 0, and turn response
   caching OFF for replicates (`--cache` omitted). Leave caching ON while
   debugging (`--cache`).
8. ProtocolQA is multiple choice with 5 options, so chance is 0.20. Report
   Cohen's kappa alongside raw agreement. Keep the raw numbers too.

## Analysis output needed (once the full study runs)

- One row per question per run.
- Between-arm disagreement (raw and kappa).
- Within-arm disagreement across the two runs (raw and kappa).
- Refusal rate per arm.
- Decomposition failure count.

## Budget

$20 hard cap. `epochs=1`. Warn before any run over $5.

## Status

- Baseline measured: single arm scores 0.467 on a 30-item slice (stderr
  0.093), vs. 0.20 chance.
- Implementation: `scaffold_study.py` (both arms, self-contained, reuses the
  `inspect_evals` LAB-Bench ProtocolQA dataset/scorer).
- **The 4-run study (single x2, chain x2, 108 items each) is DONE (2026-09-03)
  and was EXPLORATORY** — run to characterize noise and between-arm
  agreement, not to test a pre-registered hypothesis. See "Provenance" below
  before treating any of these numbers as confirmatory.
  - Logs: `...XsGEB7HRs9tat3iYpRsNVU.eval` (single, rep1), `...g9ae7mVnPWGbDZK4pZpSwv.eval`
    (single, rep2), `...bYrJvLtHfKUuEuy9bh3Auy.eval` (chain, rep1),
    `...ayiRv9qbfrbAS9Xt6896Fp.eval` (chain, rep2).
  - Accuracy: single 0.528/0.556, chain 0.491/0.528. Refusal rate: single
    0.028/0.019, chain 0.111/0.083. Within-arm kappa: single 0.588 [95% CI
    0.431, 0.733], chain 0.438 [0.284, 0.580] — difference −0.150, 95% CI
    [−0.356, 0.057] (includes 0 — not distinguishable from sampling error at
    this n). Between-arm kappa pooled: 0.411 [0.309, 0.522]. Full detail:
    `analysis/rows.csv`, `analysis/chain_refusal_detail.csv`,
    `analysis/report.txt` (via `analyze_study.py`).
  - A stale 10-item smoke-test log for `chain_arm -T replicate=1` (from
    temperature-pinning verification) also sits in `logs/`; `analyze_study.py`
    and `power_analysis.py` both auto-select the newer, full 108-item log for
    that key and warn about the duplicate, but it should be deleted for
    hygiene.

## Provenance (pre-registration)

- The 4 runs above were exploratory pilot data, not a confirmatory test.
- A power analysis (`power_analysis.py`, run 2026-09-03) used those 4 runs to
  determine how many ADDITIONAL replicates per arm a confirmatory phase would
  need to detect the observed within-arm kappa gap (0.150) at 80% power,
  alpha=0.05 two-sided.
  - **Result: 8 replicates/arm (4 independent, non-overlapping replicate
    pairs)** reach 80% power assuming the true gap equals the pilot's point
    estimate (0.150). Sensitivity: 18 replicates/arm needed if the true gap
    is only 0.10; 72 if only 0.05 — see `power_analysis.py` output for the
    full table.
  - Method: the item-level bootstrap SE of the kappa difference from the one
    observed replicate-pair (SE=0.106, n=108 items), extrapolated by
    treating additional replicate-pairs as independent, equally-noisy
    repeats and averaging (variance shrinks as 1/P, the standard result for
    averaging independent unbiased estimates). Deliberately the more
    conservative of two methods tried — a per-item multi-rater (Fleiss'
    kappa) simulation was attempted first and abandoned because it implied a
    smaller effective effect size (~0.075, half the pilot estimate), an
    artifact of resampling from only 2 pilot draws per item. See
    `power_analysis.py`'s module docstring for why.
  - The 0.150 assumed effect is itself a point estimate from a 2-replicate
    pilot whose own CI on the difference includes 0 — a power analysis
    answers "if the effect is X, how many replicates," not whether X=0.150
    is right.
- **Committed confirmatory replicate count: not yet decided.** If the user
  commits to a number, record it here with the date, and it is fixed: run
  exactly that many additional replicates per arm, analyze once, report
  whatever comes out including a null. No stopping early on a promising
  interim result, no extending the count after seeing results.

## How to run

```
# single arm, full dataset, replicate 1
inspect eval scaffold_study.py@single_arm -T replicate=1 --model anthropic/claude-haiku-4-5-20251001 --temperature 1

# single arm, replicate 2 (independent sample -- no --cache)
inspect eval scaffold_study.py@single_arm -T replicate=2 --model anthropic/claude-haiku-4-5-20251001 --temperature 1

# chain arm, replicate 1
inspect eval scaffold_study.py@chain_arm -T replicate=1 --model anthropic/claude-haiku-4-5-20251001 --temperature 1

# chain arm, replicate 2
inspect eval scaffold_study.py@chain_arm -T replicate=2 --model anthropic/claude-haiku-4-5-20251001 --temperature 1

# smoke test before a real run: small --limit, either arm
inspect eval scaffold_study.py@chain_arm --model anthropic/claude-haiku-4-5-20251001 --temperature 1 --limit 10

# view results
inspect view --log-dir logs

# analyze completed logs (agreement, kappa, refusal breakdown)
.venv/Scripts/python.exe analyze_study.py

# power analysis for planning additional replicates (no eval run)
.venv/Scripts/python.exe power_analysis.py
```

Note: `-M` is for provider-client constructor args, not sampling params — it
errors on `temperature`. Use the top-level `--temperature` flag.

- `-T replicate=N` stamps the replicate number into every sample's
  `state.store` and into the log's `task_args`, so a run is labeled in the
  log data itself -- not just distinguished by filename/timestamp.
- `-M temperature=1` is pinned explicitly rather than relying on the
  provider's default. Verified (2026-09-03): with temperature unset, Inspect
  sends no `temperature` field to Anthropic at all and the API's own default
  (1.0) applies -- not 0 -- but that's true only by absence of a setting, so
  the real runs pin it rather than depend on that.
- Add `--cache` only while debugging a single run repeatedly. Never add it
  to a run that's meant to count as one of the two independent replicates --
  it would make that replicate replay the first run's cached responses
  instead of sampling independently, collapsing the noise floor to zero.

## Per-sample bookkeeping fields (in `state.store`, both arms)

- `arm`: `"single"` or `"chain"`.
- `replicate`: the `-T replicate=N` value for this run.
- `decomposition_fallback`: chain arm only; true if the decompose step's
  output didn't parse into any sub-questions, so this item fell back to a
  single call.
- `parse_failure`: true if the final completion had no valid `ANSWER: X`
  line at all.
- `declined_insufficient_info`: true if the final answer was a valid letter,
  and that letter's choice text is "Insufficient information to answer the
  question" -- a deliberate non-answer, not a formatting failure.
- `refusal`: `parse_failure OR declined_insufficient_info`. This is the
  field rule 4 asks for, and it's defined to reconcile exactly with the
  `coverage` metric (coverage = 1 - mean(refusal)) -- confirmed 2026-09-03
  after a 10-item chain-arm smoke test showed coverage 0.900 with the old,
  narrower refusal definition (parse failures only) reporting 0/10: the
  missing item had a fully parseable `ANSWER: A` where A was, after
  shuffling, the "insufficient information" choice.
- `subquestions`, `sub_answers`: chain arm only, the decomposition and its
  fresh-context answers.
