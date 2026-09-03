"""Analysis for the scaffold-study 4-run design. See CLAUDE.md for the design.

Reads the four full-study logs (single_arm x2, chain_arm x2), found by task
name + the `replicate` task_arg (not by filename), and reports:

  - one row per (arm, replicate, sample) -> analysis/rows.csv
  - between-arm and within-arm raw agreement + Cohen's kappa, on outcome
    (correct/wrong/refusal) and separately on refusal alone
  - refusal rate per arm/replicate, decomposition failure count
  - for every chain-arm refusal: what the single arm did on that same item in
    the matching replicate, whether chain refusals repeat across the two
    chain replicates or scatter, and whether any of that item's sub-answers
    show hedging language -> analysis/chain_refusal_detail.csv

Run: .venv/Scripts/python.exe analyze_study.py
"""

import csv
import glob
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Fixed so re-running this script reproduces the same interval to the digit.
BOOTSTRAP_SEED = 20260903
N_BOOT = 5000

from inspect_ai.log import read_eval_log
from inspect_ai.scorer import CORRECT, INCORRECT, NOANSWER

LOG_DIR = Path("logs")
OUT_DIR = Path("analysis")
WANTED = {("single_arm", 1), ("single_arm", 2), ("chain_arm", 1), ("chain_arm", 2)}

OUTCOME_LABEL = {CORRECT: "correct", INCORRECT: "wrong", NOANSWER: "refusal/declined"}

# Heuristic only: sub-questions are open-ended free text with no parse
# requirement, so there is no formal "this sub-question failed" signal the
# way there is for the final answer. This regex is a best-effort scan for
# hedging language in a sub-answer, used only to distinguish "a sub-answer
# already looked uncertain" from "recombination decided the digest was
# insufficient even though the sub-answers all read as substantive."
HEDGE_RE = re.compile(
    r"(?i)\b("
    r"insufficient information|cannot (?:be )?determin\w*|unable to determine|"
    r"not enough information|no information (?:is )?(?:provided|given)|"
    r"cannot answer|unclear (?:from|based on)|"
    r"impossible to (?:determine|say|know)|"
    r"don't have (?:enough|sufficient) information|"
    r"does not (?:provide|specify|state) enough"
    r")\b"
)


def discover_logs() -> dict[tuple[str, int], tuple[str, object]]:
    """Find the one log per (arm, replicate) in WANTED; newest wins on duplicates."""
    found: dict[tuple[str, int], tuple[str, object]] = {}
    for path in sorted(glob.glob(str(LOG_DIR / "*.eval"))):
        log = read_eval_log(path)
        arm_name = log.eval.task.split("/")[-1]
        replicate = log.eval.task_args.get("replicate")
        key = (arm_name, replicate)
        if key not in WANTED:
            continue
        prior = found.get(key)
        if prior is not None:
            print(
                f"warning: multiple logs for {key}, using the newer one "
                f"({path} over {prior[0]})",
                file=sys.stderr,
            )
        found[key] = (path, log)
    missing = WANTED - found.keys()
    if missing:
        raise SystemExit(f"missing logs for: {sorted(missing)}")
    return found


def selected_choice_text(sample, score) -> str | None:
    """The literal text of whatever the model picked -- stable across runs even
    though choice *order* is reshuffled independently in every run."""
    letter = (score.answer or "").strip()
    if len(letter) != 1:
        return None
    idx = ord(letter.upper()) - ord("A")
    return sample.choices[idx] if 0 <= idx < len(sample.choices) else None


def build_rows(found) -> list[dict]:
    rows = []
    for (arm_name, replicate), (_path, log) in found.items():
        arm = "single" if arm_name == "single_arm" else "chain"
        for s in log.samples:
            score = s.scores["precision_choice"]
            rows.append(
                {
                    "arm": arm,
                    "replicate": replicate,
                    "sample_id": s.id,
                    "outcome": score.value,
                    "chosen_choice_text": selected_choice_text(s, score),
                    "refusal": bool(s.store.get("refusal")),
                    "parse_failure": bool(s.store.get("parse_failure")),
                    "declined_insufficient_info": bool(
                        s.store.get("declined_insufficient_info")
                    ),
                    "decomposition_fallback": bool(
                        s.store.get("decomposition_fallback", False)
                    ),
                    "num_subquestions": len(s.store.get("subquestions", []) or []),
                    "subquestions": s.store.get("subquestions", []) or [],
                    "sub_answers": s.store.get("sub_answers", []) or [],
                    "final_completion": s.output.completion,
                }
            )
    return rows


def write_rows_csv(rows: list[dict], path: Path) -> None:
    fields = [
        "arm",
        "replicate",
        "sample_id",
        "outcome",
        "chosen_choice_text",
        "refusal",
        "parse_failure",
        "declined_insufficient_info",
        "decomposition_fallback",
        "num_subquestions",
        "subquestions",
        "sub_answers",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            row = dict(r)
            row["subquestions"] = json.dumps(row["subquestions"], ensure_ascii=False)
            row["sub_answers"] = json.dumps(row["sub_answers"], ensure_ascii=False)
            w.writerow(row)


def index_rows(rows: list[dict]) -> dict:
    idx: dict = defaultdict(dict)
    for r in rows:
        idx[(r["arm"], r["replicate"])][r["sample_id"]] = r
    return idx


def cohens_kappa(pairs: list[tuple]) -> tuple[float | None, float | None, int]:
    """pairs: list of (rater_a_label, rater_b_label) for the same items in the
    same order. Returns (raw_agreement, kappa, n)."""
    n = len(pairs)
    if n == 0:
        return None, None, 0
    agree = sum(1 for a, b in pairs if a == b)
    raw = agree / n
    labels = sorted({a for a, _ in pairs} | {b for _, b in pairs})
    a_counts = Counter(a for a, _ in pairs)
    b_counts = Counter(b for _, b in pairs)
    pe = sum((a_counts[label] / n) * (b_counts[label] / n) for label in labels)
    kappa = float("nan") if pe >= 1.0 else (raw - pe) / (1 - pe)
    return raw, kappa, n


def paired(idx, key_a, key_b, field: str):
    a_map, b_map = idx[key_a], idx[key_b]
    common = sorted(set(a_map) & set(b_map))
    if len(common) != len(a_map) or len(common) != len(b_map):
        print(
            f"warning: {key_a} has {len(a_map)} rows, {key_b} has {len(b_map)}, "
            f"only {len(common)} sample ids in common",
            file=sys.stderr,
        )
    return [(a_map[i][field], b_map[i][field]) for i in common], common


def fmt_kappa(raw, kappa, n) -> str:
    return f"n={n}  raw agreement={raw:.3f}  kappa={kappa:.3f}"


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion -- better than the
    normal approximation at small n / extreme proportions, both of which
    apply to refusal counts like 2/108 or 12/108."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * ((p * (1 - p) / n) + z**2 / (4 * n**2)) ** 0.5 / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def bootstrap_paired_two_kappas(
    pairs_a: list[tuple], pairs_b: list[tuple], n_boot: int = N_BOOT, seed: int = BOOTSTRAP_SEED
):
    """Item-level paired bootstrap: pairs_a[i] and pairs_b[i] must describe the
    SAME underlying item i (e.g. single-arm rep1-vs-rep2 outcome for item i,
    chain-arm rep1-vs-rep2 outcome for that same item i). Each bootstrap draw
    resamples item indices once and applies that same resample to both lists,
    so ka_boot[j] and kb_boot[j] come from matched data -- this is what lets
    the difference kb_boot[j]-ka_boot[j] isolate "is arm B less reproducible
    than arm A on these items" from ordinary sampling noise.

    Returns (ka_boot, kb_boot, diff_boot), each a sorted-later raw list.
    """
    assert len(pairs_a) == len(pairs_b)
    n = len(pairs_a)
    rng = random.Random(seed)
    ka_boot, kb_boot, diff_boot = [], [], []
    for _ in range(n_boot):
        idxs = [rng.randrange(n) for _ in range(n)]
        ra = [pairs_a[i] for i in idxs]
        rb = [pairs_b[i] for i in idxs]
        _, ka, _ = cohens_kappa(ra)
        _, kb, _ = cohens_kappa(rb)
        if ka == ka and kb == kb:  # drop NaN (only possible if a resample is degenerate)
            ka_boot.append(ka)
            kb_boot.append(kb)
            diff_boot.append(kb - ka)
    return ka_boot, kb_boot, diff_boot


def bootstrap_kappa(pairs: list[tuple], n_boot: int = N_BOOT, seed: int = BOOTSTRAP_SEED):
    """Independent-sample bootstrap CI for a single kappa (no pairing to another arm)."""
    n = len(pairs)
    rng = random.Random(seed)
    boot = []
    for _ in range(n_boot):
        idxs = [rng.randrange(n) for _ in range(n)]
        resampled = [pairs[i] for i in idxs]
        _, k, _ = cohens_kappa(resampled)
        if k == k:
            boot.append(k)
    return boot


def percentile_ci(values: list[float], alpha: float = 0.05) -> tuple[float, float]:
    v = sorted(values)
    n = len(v)
    lo = v[max(0, int(alpha / 2 * n))]
    hi = v[min(n - 1, int((1 - alpha / 2) * n) - 1)]
    return lo, hi


def paired_excluding_refusals(idx, key_a, key_b, field):
    a_map, b_map = idx[key_a], idx[key_b]
    common = sorted(set(a_map) & set(b_map))
    kept = [i for i in common if not a_map[i]["refusal"] and not b_map[i]["refusal"]]
    return [(a_map[i][field], b_map[i][field]) for i in kept], kept


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    found = discover_logs()
    rows = build_rows(found)
    write_rows_csv(rows, OUT_DIR / "rows.csv")
    idx = index_rows(rows)

    report = []

    def emit(line: str = "") -> None:
        report.append(line)
        print(line)

    emit("=" * 70)
    emit("BETWEEN-ARM AGREEMENT (single vs chain, matched by replicate)")
    emit("=" * 70)
    pooled_outcome, pooled_refusal = [], []
    for rep in (1, 2):
        pairs, _ = paired(idx, ("single", rep), ("chain", rep), "outcome")
        raw, kappa, n = cohens_kappa(pairs)
        emit(f"replicate {rep}, outcome (correct/wrong/refusal): {fmt_kappa(raw, kappa, n)}")
        pooled_outcome += pairs

        rpairs, _ = paired(idx, ("single", rep), ("chain", rep), "refusal")
        raw, kappa, n = cohens_kappa(rpairs)
        emit(f"replicate {rep}, refusal (yes/no):                {fmt_kappa(raw, kappa, n)}")
        pooled_refusal += rpairs
    raw, kappa, n = cohens_kappa(pooled_outcome)
    emit(f"pooled across replicates, outcome:                    {fmt_kappa(raw, kappa, n)}")
    raw, kappa, n = cohens_kappa(pooled_refusal)
    emit(f"pooled across replicates, refusal:                    {fmt_kappa(raw, kappa, n)}")

    emit("")
    emit("=" * 70)
    emit("WITHIN-ARM AGREEMENT (replicate 1 vs replicate 2, same arm) -- noise floor")
    emit("=" * 70)
    within_arm_outcome_pairs = {}
    for arm in ("single", "chain"):
        pairs, _ = paired(idx, (arm, 1), (arm, 2), "outcome")
        within_arm_outcome_pairs[arm] = pairs
        raw, kappa, n = cohens_kappa(pairs)
        emit(f"{arm:>6} arm, outcome (correct/wrong/refusal): {fmt_kappa(raw, kappa, n)}")
        rpairs, _ = paired(idx, (arm, 1), (arm, 2), "refusal")
        raw, kappa, n = cohens_kappa(rpairs)
        emit(f"{arm:>6} arm, refusal (yes/no):                {fmt_kappa(raw, kappa, n)}")

    emit("")
    emit("=" * 70)
    emit("IS THE WITHIN-ARM KAPPA GAP REAL, OR SAMPLING ERROR? (item-paired bootstrap, "
         f"n_boot={N_BOOT}, seed={BOOTSTRAP_SEED})")
    emit("=" * 70)
    emit("The two arms cover the same 108 items, so each bootstrap draw resamples item\n"
         "indices ONCE and applies that same resample to both arms -- this lets the\n"
         "difference isolate 'chain is less reproducible than single' from ordinary\n"
         "item-sampling noise, rather than comparing two independently-noisy intervals.")
    ka_boot, kb_boot, diff_boot = bootstrap_paired_two_kappas(
        within_arm_outcome_pairs["single"], within_arm_outcome_pairs["chain"]
    )
    _, single_kappa_point, n_within = cohens_kappa(within_arm_outcome_pairs["single"])
    _, chain_kappa_point, _ = cohens_kappa(within_arm_outcome_pairs["chain"])
    single_lo, single_hi = percentile_ci(ka_boot)
    chain_lo, chain_hi = percentile_ci(kb_boot)
    diff_lo, diff_hi = percentile_ci(diff_boot)
    diff_point = chain_kappa_point - single_kappa_point
    emit(f"single within-arm kappa: {single_kappa_point:.3f}  95% CI [{single_lo:.3f}, {single_hi:.3f}]")
    emit(f" chain within-arm kappa: {chain_kappa_point:.3f}  95% CI [{chain_lo:.3f}, {chain_hi:.3f}]")
    emit(f"difference (chain - single): {diff_point:.3f}  95% CI [{diff_lo:.3f}, {diff_hi:.3f}]")
    verdict = "excludes 0 -> distinguishable at 95%" if diff_lo > 0 or diff_hi < 0 else "includes 0 -> NOT distinguishable from sampling error at 95%"
    emit(f"-> {verdict}")

    emit("")
    emit("=" * 70)
    emit("DOES THE GAP SURVIVE EXCLUDING REFUSALS? (items both replicates answered)")
    emit("=" * 70)
    emit("Restricting each arm to items where NEITHER replicate refused. This changes\n"
         "the item set independently per arm (chain excludes more items, since it\n"
         "refuses more) -- so unlike the paired test above, the two kappas below no\n"
         "longer describe the same items, and the difference CI is an ordinary\n"
         "independent-samples bootstrap, not a matched-pairs one.")
    filtered_kappa = {}
    for arm in ("single", "chain"):
        fpairs, fkept = paired_excluding_refusals(idx, (arm, 1), (arm, 2), "outcome")
        raw, kappa, n = cohens_kappa(fpairs)
        boot = bootstrap_kappa(fpairs)
        lo, hi = percentile_ci(boot)
        filtered_kappa[arm] = (kappa, lo, hi, n, boot)
        excluded = 108 - n
        emit(f"{arm:>6} arm: {fmt_kappa(raw, kappa, n)}  95% CI [{lo:.3f}, {hi:.3f}]  "
             f"({excluded} items excluded for a refusal in either replicate)")
    n_common = min(len(filtered_kappa["single"][4]), len(filtered_kappa["chain"][4]))
    diff_boot_f = [
        filtered_kappa["chain"][4][j] - filtered_kappa["single"][4][j] for j in range(n_common)
    ]
    diff_point_f = filtered_kappa["chain"][0] - filtered_kappa["single"][0]
    dlo_f, dhi_f = percentile_ci(diff_boot_f)
    emit(f"difference (chain - single) on refusal-free items: {diff_point_f:.3f}  "
         f"95% CI [{dlo_f:.3f}, {dhi_f:.3f}]  (independent-samples bootstrap)")
    verdict_f = ("excludes 0 -> gap persists without refusals"
                 if dlo_f > 0 or dhi_f < 0
                 else "includes 0 -> gap not distinguishable from sampling error once refusals are excluded")
    emit(f"-> {verdict_f}")

    emit("")
    emit("=" * 70)
    emit("REFUSAL RATE AND DECOMPOSITION FAILURES")
    emit("=" * 70)
    for (arm_name, replicate), (_p, log) in sorted(found.items()):
        arm = "single" if arm_name == "single_arm" else "chain"
        arm_rows = idx[(arm, replicate)]
        n = len(arm_rows)
        refusals = sum(1 for r in arm_rows.values() if r["refusal"])
        fallbacks = sum(1 for r in arm_rows.values() if r["decomposition_fallback"])
        emit(
            f"{arm:>6} arm, replicate {replicate}: refusals {refusals}/{n} "
            f"({refusals / n:.1%})   decomposition fallbacks {fallbacks}/{n}"
        )

    emit("")
    emit("=" * 70)
    emit("CHAIN-ARM REFUSALS: what did the single arm do on the same item?")
    emit("=" * 70)
    detail_rows = []
    for rep in (1, 2):
        chain_rows = idx[("chain", rep)]
        single_rows = idx[("single", rep)]
        refused_ids = sorted(i for i, r in chain_rows.items() if r["refusal"])
        crosstab = Counter()
        for sid in refused_ids:
            single_row = single_rows.get(sid)
            single_outcome = OUTCOME_LABEL.get(single_row["outcome"], "?") if single_row else "missing"
            crosstab[single_outcome] += 1

            chain_row = chain_rows[sid]
            hedged_idx = [
                j
                for j, ans in enumerate(chain_row["sub_answers"])
                if HEDGE_RE.search(ans)
            ]
            detail_rows.append(
                {
                    "replicate": rep,
                    "sample_id": sid,
                    "single_arm_outcome_same_item": single_outcome,
                    "num_subquestions": chain_row["num_subquestions"],
                    "any_subanswer_hedged": bool(hedged_idx),
                    "hedged_subquestion_indices": json.dumps(hedged_idx),
                    "final_completion_tail": chain_row["final_completion"][-400:],
                }
            )
        emit(f"replicate {rep}: chain refused on {len(refused_ids)}/108 items. "
             f"On those same items, single arm was:")
        for label in ("correct", "wrong", "refusal/declined", "missing"):
            if crosstab.get(label):
                emit(f"    {label}: {crosstab[label]}")

    with open(OUT_DIR / "chain_refusal_detail.csv", "w", newline="", encoding="utf-8") as f:
        fields = [
            "replicate",
            "sample_id",
            "single_arm_outcome_same_item",
            "num_subquestions",
            "any_subanswer_hedged",
            "hedged_subquestion_indices",
            "final_completion_tail",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(detail_rows)

    emit("")
    emit("=" * 70)
    emit("ARE CHAIN-ARM REFUSALS THE SAME ITEMS ACROSS REPLICATES, OR SCATTERED?")
    emit("=" * 70)
    refused_r1 = {i for i, r in idx[("chain", 1)].items() if r["refusal"]}
    refused_r2 = {i for i, r in idx[("chain", 2)].items() if r["refusal"]}
    overlap = refused_r1 & refused_r2
    union = refused_r1 | refused_r2
    emit(f"replicate 1 refused: {len(refused_r1)}   replicate 2 refused: {len(refused_r2)}")
    emit(f"refused in BOTH replicates (same item, same run's independent sample): {len(overlap)}")
    emit(f"refused in EITHER replicate (union): {len(union)}")
    if union:
        emit(f"overlap / union (Jaccard): {len(overlap) / len(union):.3f}")
    if refused_r1:
        emit(f"of replicate 1's refusals, fraction that recur in replicate 2: "
             f"{len(overlap) / len(refused_r1):.3f}")
    if refused_r2:
        emit(f"of replicate 2's refusals, fraction that recur in replicate 1: "
             f"{len(overlap) / len(refused_r2):.3f}")

    emit("")
    emit("=" * 70)
    emit("DOES THE REFUSAL COME FROM A SUB-QUESTION OR FROM RECOMBINATION?")
    emit("=" * 70)
    emit(
        "Caveat: sub-questions are free-text with no pass/fail parsing, so there is\n"
        "no formal 'this sub-question refused' signal to read off directly. This is a\n"
        "heuristic: does at least one of the item's stored sub-answers contain\n"
        "hedging language (\"insufficient information\", \"cannot determine\", etc.)?\n"
        "'hedged' means a sub-answer already looked uncertain before recombination;\n"
        "'clean' means all sub-answers read as substantive and the decline happened\n"
        "only at the final, recombination step."
    )
    hedged_count = sum(1 for d in detail_rows if d["any_subanswer_hedged"])
    clean_count = len(detail_rows) - hedged_count
    emit(f"across both replicates' {len(detail_rows)} chain refusals: "
         f"{hedged_count} had a hedged sub-answer, {clean_count} were clean "
         f"(recombination-only decline)")
    for rep in (1, 2):
        rep_detail = [d for d in detail_rows if d["replicate"] == rep]
        h = sum(1 for d in rep_detail if d["any_subanswer_hedged"])
        emit(f"  replicate {rep}: {h}/{len(rep_detail)} hedged")

    emit("")
    emit("=" * 70)
    emit("SUMMARY TABLE (for the writeup)")
    emit("=" * 70)

    emit("")
    emit("Per run:")
    emit("")
    emit("| Arm | Rep | Accuracy, all items (95% CI) [n=108] | Accuracy, answered-only (precision metric) [n=answered] | Refusal rate (95% CI) |")
    emit("|---|---|---|---|---|")
    for (arm_name, replicate), (_p, log) in sorted(found.items()):
        arm = "single" if arm_name == "single_arm" else "chain"
        sc = log.results.scores[0]
        acc = sc.metrics["accuracy"].value
        acc_se = sc.metrics["stderr"].value
        acc_lo, acc_hi = acc - 1.96 * acc_se, acc + 1.96 * acc_se
        precision = sc.metrics["precision"].value
        n = len(idx[(arm, replicate)])
        refusals = sum(1 for r in idx[(arm, replicate)].values() if r["refusal"])
        answered = n - refusals
        ref_rate = refusals / n
        ref_lo, ref_hi = wilson_ci(refusals, n)
        emit(
            f"| {arm} | {replicate} | {acc:.3f} [{acc_lo:.3f}, {acc_hi:.3f}] | "
            f"{precision:.3f} (n={answered}) | "
            f"{ref_rate:.3f} ({refusals}/{n}) [{ref_lo:.3f}, {ref_hi:.3f}] |"
        )

    emit("")
    emit("Agreement (Cohen's kappa, item-level, on correct/wrong/refusal outcome):")
    emit("")
    emit("| Comparison | n | kappa | 95% CI |")
    emit("|---|---|---|---|")
    emit(f"| Within-arm, single (rep1 vs rep2) | {n_within} | {single_kappa_point:.3f} | "
         f"[{single_lo:.3f}, {single_hi:.3f}] |")
    emit(f"| Within-arm, chain (rep1 vs rep2) | {n_within} | {chain_kappa_point:.3f} | "
         f"[{chain_lo:.3f}, {chain_hi:.3f}] |")
    emit(f"| Within-arm difference (chain - single) | {n_within} | {diff_point:.3f} | "
         f"[{diff_lo:.3f}, {diff_hi:.3f}] |")
    emit(f"| Within-arm, single, refusal-free items only | {filtered_kappa['single'][3]} | "
         f"{filtered_kappa['single'][0]:.3f} | [{filtered_kappa['single'][1]:.3f}, "
         f"{filtered_kappa['single'][2]:.3f}] |")
    emit(f"| Within-arm, chain, refusal-free items only | {filtered_kappa['chain'][3]} | "
         f"{filtered_kappa['chain'][0]:.3f} | [{filtered_kappa['chain'][1]:.3f}, "
         f"{filtered_kappa['chain'][2]:.3f}] |")
    emit(f"| Within-arm difference, refusal-free (chain - single) | n varies | "
         f"{diff_point_f:.3f} | [{dlo_f:.3f}, {dhi_f:.3f}] |")

    between_pooled_pairs = []
    for rep in (1, 2):
        bpairs, _ = paired(idx, ("single", rep), ("chain", rep), "outcome")
        _, bkappa, bn = cohens_kappa(bpairs)
        bboot = bootstrap_kappa(bpairs)
        blo, bhi = percentile_ci(bboot)
        emit(f"| Between-arm, replicate {rep} | {bn} | {bkappa:.3f} | [{blo:.3f}, {bhi:.3f}] |")
        between_pooled_pairs += bpairs
    _, bkappa_pooled, bn_pooled = cohens_kappa(between_pooled_pairs)
    bboot_pooled = bootstrap_kappa(between_pooled_pairs)
    blo_p, bhi_p = percentile_ci(bboot_pooled)
    emit(f"| Between-arm, pooled | {bn_pooled} | {bkappa_pooled:.3f} | "
         f"[{blo_p:.3f}, {bhi_p:.3f}] |")
    emit("")
    emit(f"(All intervals: percentile bootstrap, n_boot={N_BOOT}, seed={BOOTSTRAP_SEED}, "
         "except accuracy which uses Inspect's own stderr x 1.96, and refusal rate "
         "which uses a Wilson score interval. Answered-only accuracy (the precision "
         "metric) is reported as a point estimate with no interval.)")

    emit("")
    emit(f"Wrote {OUT_DIR / 'rows.csv'} ({len(rows)} rows) and "
         f"{OUT_DIR / 'chain_refusal_detail.csv'} ({len(detail_rows)} rows).")

    (OUT_DIR / "report.txt").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
