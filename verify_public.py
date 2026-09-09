"""Recompute the published statistics from public/rows.csv alone.

Standalone verification: no inspect_ai import, no logs/ or analysis/
directory (both are deliberately absent from this repo -- see README.md),
stdlib only. Reads public/rows.csv, which stores its two boolean columns
(`refusal`, and implicitly `parse_failure` / `declined_insufficient_info`
which this script doesn't need) as the literal strings "True"/"False", not
native CSV booleans.

The functions below (cohens_kappa, wilson_ci, bootstrap_paired_two_kappas,
bootstrap_kappa, percentile_ci) are copied, not imported, from
analyze_study.py so this script has no dependency on that module or on
inspect_ai being installed.

Reproduces and prints: per-run outcome counts, accuracy, precision
(correct/answered), coverage (answered/108), abstention rate with Wilson
intervals, within-arm and between-arm Cohen's kappa, the paired bootstrap
kappa difference and CI (seed 20260903, n_boot 5000), and the chain-arm
abstention Jaccard across replicates. Asserts each against known-good
values from the original analyze_study.py run and prints PASS/FAIL.

Run: python verify_public.py
"""

import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

BOOTSTRAP_SEED = 20260903
N_BOOT = 5000

ROWS_CSV = Path("public/rows.csv")


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


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
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
    SAME underlying item i. Each bootstrap draw resamples item indices once
    and applies that same resample to both lists.

    Returns (ka_boot, kb_boot, diff_boot).
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
        if ka == ka and kb == kb:  # drop NaN
            ka_boot.append(ka)
            kb_boot.append(kb)
            diff_boot.append(kb - ka)
    return ka_boot, kb_boot, diff_boot


def bootstrap_kappa(pairs: list[tuple], n_boot: int = N_BOOT, seed: int = BOOTSTRAP_SEED):
    """Independent-sample bootstrap CI for a single kappa."""
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


def load_rows() -> list[dict]:
    with open(ROWS_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["replicate"] = int(r["replicate"])
        r["refusal"] = r["refusal"] == "True"
    return rows


def index_rows(rows: list[dict]) -> dict:
    idx: dict = defaultdict(dict)
    for r in rows:
        idx[(r["arm"], r["replicate"])][r["sample_id"]] = r
    return idx


def paired(idx, key_a, key_b, field: str):
    a_map, b_map = idx[key_a], idx[key_b]
    common = sorted(set(a_map) & set(b_map))
    return [(a_map[i][field], b_map[i][field]) for i in common], common


def paired_excluding_refusals(idx, key_a, key_b, field):
    a_map, b_map = idx[key_a], idx[key_b]
    common = sorted(set(a_map) & set(b_map))
    kept = [i for i in common if not a_map[i]["refusal"] and not b_map[i]["refusal"]]
    return [(a_map[i][field], b_map[i][field]) for i in kept], kept


CHECKS: list[tuple[str, bool]] = []


def check(label: str, actual, expected, tol: float = 0.0005) -> None:
    if isinstance(expected, tuple) or isinstance(expected, int):
        ok = actual == expected
    else:
        ok = abs(actual - expected) <= tol
    CHECKS.append((f"{label}: got {actual}, expected {expected}", ok))


def main() -> None:
    rows = load_rows()
    idx = index_rows(rows)

    print("=" * 70)
    print("PER-RUN OUTCOME COUNTS, ACCURACY, PRECISION, COVERAGE, ABSTENTION")
    print("=" * 70)
    expected_counts = {
        ("single", 1): (57, 48, 3),
        ("single", 2): (60, 46, 2),
        ("chain", 1): (53, 43, 12),
        ("chain", 2): (57, 42, 9),
    }
    for (arm, rep), (exp_c, exp_i, exp_n) in expected_counts.items():
        arm_rows = idx[(arm, rep)]
        n = len(arm_rows)
        counts = Counter(r["outcome"] for r in arm_rows.values())
        c, i, no = counts.get("C", 0), counts.get("I", 0), counts.get("N", 0)
        refusals = sum(1 for r in arm_rows.values() if r["refusal"])
        answered = n - refusals
        accuracy = c / n
        precision = c / answered if answered else float("nan")
        coverage = answered / n
        abst_rate = refusals / n
        abst_lo, abst_hi = wilson_ci(refusals, n)
        print(
            f"{arm:>6} rep{rep}: C/I/N = {c}/{i}/{no}  accuracy={accuracy:.3f}  "
            f"precision={precision:.3f}  coverage={coverage:.3f}  "
            f"abstention={abst_rate:.3f} [{abst_lo:.3f}, {abst_hi:.3f}]"
        )
        check(f"{arm} rep{rep} C/I/N", (c, i, no), (exp_c, exp_i, exp_n))

    print()
    print("=" * 70)
    print("WITHIN-ARM COHEN'S KAPPA (replicate 1 vs replicate 2)")
    print("=" * 70)
    within = {}
    for arm in ("single", "chain"):
        pairs, _ = paired(idx, (arm, 1), (arm, 2), "outcome")
        within[arm] = pairs
        raw, kappa, n = cohens_kappa(pairs)
        print(f"{arm:>6} arm: n={n}  raw agreement={raw:.3f}  kappa={kappa:.3f}")

    ka_boot, kb_boot, diff_boot = bootstrap_paired_two_kappas(within["single"], within["chain"])
    _, k_single, n_within = cohens_kappa(within["single"])
    _, k_chain, _ = cohens_kappa(within["chain"])
    diff_point = k_chain - k_single
    diff_lo, diff_hi = percentile_ci(diff_boot)
    print(f"difference (chain - single): {diff_point:.3f}  95% CI [{diff_lo:.3f}, {diff_hi:.3f}]")
    check("within-arm kappa single", k_single, 0.588, tol=0.001)
    check("within-arm kappa chain", k_chain, 0.438, tol=0.001)
    check("within-arm kappa difference", diff_point, -0.150, tol=0.001)
    check("within-arm kappa difference CI lo", diff_lo, -0.356, tol=0.001)
    check("within-arm kappa difference CI hi", diff_hi, 0.057, tol=0.001)

    print()
    print("=" * 70)
    print("WITHIN-ARM KAPPA, REFUSAL-FREE ITEMS ONLY")
    print("=" * 70)
    filtered = {}
    for arm in ("single", "chain"):
        fpairs, fkept = paired_excluding_refusals(idx, (arm, 1), (arm, 2), "outcome")
        raw, kappa, n = cohens_kappa(fpairs)
        boot = bootstrap_kappa(fpairs)
        lo, hi = percentile_ci(boot)
        filtered[arm] = (kappa, lo, hi, n, boot)
        print(f"{arm:>6} arm: n={n}  kappa={kappa:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
    n_common = min(len(filtered["single"][4]), len(filtered["chain"][4]))
    diff_boot_f = [filtered["chain"][4][j] - filtered["single"][4][j] for j in range(n_common)]
    diff_point_f = filtered["chain"][0] - filtered["single"][0]
    dlo_f, dhi_f = percentile_ci(diff_boot_f)
    print(f"difference (chain - single), refusal-free: {diff_point_f:.3f}  "
          f"95% CI [{dlo_f:.3f}, {dhi_f:.3f}]")
    check("refusal-free kappa single", filtered["single"][0], 0.610, tol=0.001)
    check("refusal-free kappa chain", filtered["chain"][0], 0.507, tol=0.001)
    check("refusal-free kappa difference", diff_point_f, -0.104, tol=0.001)
    check("refusal-free kappa difference CI lo", dlo_f, -0.345, tol=0.001)
    check("refusal-free kappa difference CI hi", dhi_f, 0.142, tol=0.001)

    print()
    print("=" * 70)
    print("BETWEEN-ARM COHEN'S KAPPA, POOLED ACROSS REPLICATES")
    print("=" * 70)
    between_pooled = []
    for rep in (1, 2):
        bpairs, _ = paired(idx, ("single", rep), ("chain", rep), "outcome")
        _, bkappa, bn = cohens_kappa(bpairs)
        print(f"replicate {rep}: n={bn}  kappa={bkappa:.3f}")
        between_pooled += bpairs
    _, bkappa_pooled, bn_pooled = cohens_kappa(between_pooled)
    print(f"pooled: n={bn_pooled}  kappa={bkappa_pooled:.3f}")
    check("between-arm pooled kappa", bkappa_pooled, 0.411, tol=0.001)

    print()
    print("=" * 70)
    print("CHAIN-ARM ABSTENTION JACCARD ACROSS REPLICATES")
    print("=" * 70)
    refused_r1 = {i for i, r in idx[("chain", 1)].items() if r["refusal"]}
    refused_r2 = {i for i, r in idx[("chain", 2)].items() if r["refusal"]}
    overlap = refused_r1 & refused_r2
    union = refused_r1 | refused_r2
    jaccard = len(overlap) / len(union)
    print(f"replicate 1 abstained: {len(refused_r1)}   replicate 2 abstained: {len(refused_r2)}")
    print(f"overlap / union (Jaccard): {jaccard:.3f}")
    check("chain abstention Jaccard", jaccard, 0.235, tol=0.001)

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    failed = [msg for msg, ok in CHECKS if not ok]
    for msg, ok in CHECKS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
    if failed:
        print(f"\nFAIL: {len(failed)}/{len(CHECKS)} checks failed.")
        sys.exit(1)
    else:
        print(f"\nPASS: all {len(CHECKS)} checks matched known-good values.")


if __name__ == "__main__":
    main()
