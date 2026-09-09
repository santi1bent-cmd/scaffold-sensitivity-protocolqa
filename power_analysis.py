"""Power analysis: how many replicates per arm to detect the observed within-arm
kappa gap (single 0.588 vs chain 0.438, diff = 0.150) at 80% power?

Planning only -- reads the four existing pilot logs and computes from them.
Does NOT call the model or run any new eval.

Method
------
A first attempt simulated "R replicates" by, for each item, resampling with
replacement from that item's two pilot outcomes (a per-item Fleiss'-kappa
generalization). That was abandoned: even at R=2 it doesn't reproduce the
observed kappa gap in expectation (its long-run mean gap came out ~0.075,
half the pilot's 0.150) because independently coin-flipping between two
observed values gives an item that disagreed in the pilot a 50% chance of
*coincidentally agreeing* on replay -- something the real 2-draw pilot never
showed. Fixing that requires assuming a per-item true probability the 2-draw
pilot cannot actually identify, which trades one debatable assumption for
another rather than removing it.

Instead: treat each additional independent replicate-pair (replicate 3 vs 4,
5 vs 6, ...) as an independent repeat of the SAME kind of comparison we
already ran once (replicate 1 vs 2), with the SAME noise level. This needs
only one assumption -- that a future pair of independent-sampling runs is
about as noisy as the one pair we've already observed -- and no per-item
outcome model. Averaging P independent, equally-noisy estimates reduces
variance by a factor of P (the standard result for averaging independent
unbiased estimators): SE_diff(P pairs) = SE_2 / sqrt(P), where SE_2 is the
already-computed item-level bootstrap SE of the kappa difference from our
one real replicate-pair. R = 2 * P total replicates per arm.

Caveat: SE_2 (se_one_pair below) comes from bootstrap_paired_two_kappas,
which resamples ITEMS while holding each item's observed replicate-1/
replicate-2 outcome pair fixed. That's an item-level SE from the one
replicate-pair we have; it says nothing about how much noisier or quieter
a *different* replicate-pair would be, so SE_2/sqrt(P) does not actually
model the variance reduction from collecting P independent replicate-pairs
-- it's a lower bound on that SE, not an estimate of it, because it carries
no information about pair-to-pair variability. The resulting replicate
counts (8/18/72 below) are therefore a floor on what a confirmatory phase
would need, not a conservative (i.e. safely-more-than-enough) estimate.
"""

import math

from analyze_study import bootstrap_paired_two_kappas, build_rows, cohens_kappa, discover_logs, index_rows, paired

ALPHA = 0.05
TARGET_POWER = 0.80
Z_CRIT = 1.959963985  # two-sided normal critical value at alpha=0.05
Z_BETA = 0.8416212336  # one-sided normal critical value at power=0.80


def normal_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / 2**0.5))


def power_for_se(se: float, delta: float) -> float:
    z = delta / se
    return normal_cdf(z - Z_CRIT) + normal_cdf(-z - Z_CRIT)


def pairs_needed(se_one_pair: float, delta: float, target_power: float = TARGET_POWER) -> int:
    """Minimum P (independent replicate-pairs) so SE_2/sqrt(P) reaches the SE
    needed for target_power against effect size delta, at alpha=0.05 two-sided."""
    assert target_power == TARGET_POWER, "Z_BETA is hardcoded for 80% power"
    required_se = delta / (Z_CRIT + Z_BETA)
    p = (se_one_pair / required_se) ** 2
    return max(1, math.ceil(p))


def main() -> None:
    found = discover_logs()
    rows = build_rows(found)
    idx = index_rows(rows)

    single_pairs, _ = paired(idx, ("single", 1), ("single", 2), "outcome")
    chain_pairs, _ = paired(idx, ("chain", 1), ("chain", 2), "outcome")

    _, k_single, n = cohens_kappa(single_pairs)
    _, k_chain, _ = cohens_kappa(chain_pairs)
    observed_diff = k_chain - k_single

    ka_boot, kb_boot, diff_boot = bootstrap_paired_two_kappas(single_pairs, chain_pairs)
    mean_diff = sum(diff_boot) / len(diff_boot)
    se_2 = (sum((d - mean_diff) ** 2 for d in diff_boot) / (len(diff_boot) - 1)) ** 0.5

    print("=" * 70)
    print("INPUTS (from the 4 pilot runs already completed)")
    print("=" * 70)
    print(f"single within-arm kappa (R=2): {k_single:.3f}")
    print(f" chain within-arm kappa (R=2): {k_chain:.3f}")
    print(f"observed difference (chain - single): {observed_diff:.3f}")
    print(f"bootstrap SE of that difference, from ONE replicate-pair (n={n} items): {se_2:.4f}")

    print()
    print("=" * 70)
    print(f"REPLICATES NEEDED FOR {TARGET_POWER:.0%} POWER (alpha=0.05, two-sided)")
    print("=" * 70)
    print(f"{'assumed true diff':>20} {'pairs needed (P)':>18} {'replicates/arm (R=2P)':>22} {'power at that R':>16}")
    for label, delta in (("0.15 (pilot estimate)", abs(observed_diff)), ("0.10 (conservative)", 0.10), ("0.05 (very conservative)", 0.05)):
        p = pairs_needed(se_2, abs(delta))
        r = 2 * p
        se_at_p = se_2 / (p**0.5)
        power = power_for_se(se_at_p, abs(delta))
        print(f"{label:>20} {p:>18} {r:>22} {power:>16.3f}")

    print()
    print("Headline: assuming the true gap really is the pilot's 0.15, "
          f"{2 * pairs_needed(se_2, abs(observed_diff))} replicates per arm "
          f"({pairs_needed(se_2, abs(observed_diff))} independent pairs) reach {TARGET_POWER:.0%} power.")
    print()
    print("Caveats:")
    print("- The 0.15 assumed effect is itself a point estimate from a 2-replicate pilot")
    print("  whose own CI on the difference was [-0.356, 0.057] -- consistent with zero.")
    print("  A power analysis answers 'if the effect is X, how many replicates' -- it")
    print("  cannot tell you whether X=0.15 is right. The 0.10 and 0.05 rows above show")
    print("  how much the required count grows if the true effect is smaller.")
    print("- This method assumes each future replicate-pair is about as noisy as the one")
    print("  pair already observed. It is method-conservative (see module docstring) --")
    print("  a more sample-efficient multi-rater estimator (Fleiss' kappa across all R")
    print("  replicates at once, rather than averaging disjoint pairs) would likely need")
    print("  somewhat fewer replicates, but requires a per-item outcome model this")
    print("  2-replicate pilot cannot support without introducing its own bias.")
    print("- R must be even for this pairing method to use every replicate; an odd R")
    print("  wastes one replicate for THIS projection, though the eventual real analysis")
    print("  can still use every replicate jointly once collected.")
    print("- SE_2 is an item-level bootstrap SE from the ONE replicate-pair we have; it")
    print("  carries no information about pair-to-pair variability, so SE_2/sqrt(P) is a")
    print("  lower bound on the SE of P replicate-pairs, not an estimate of it. The 8/18/72")
    print("  replicate counts above are a floor, not a conservative (extra-safe) estimate.")


if __name__ == "__main__":
    main()
