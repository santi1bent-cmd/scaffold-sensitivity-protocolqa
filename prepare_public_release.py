"""Produce the sanitized files that go in the public repo, under public/.

Strips every column that could carry free text (sub-question text, model
completions, chosen-answer text) out of the analysis CSVs, keeping only
categorical/numeric fields (arm, replicate, sample_id, outcome, refusal
flags, counts). analysis/report.txt is already aggregate-only (it only ever
printed counts and kappa values, never raw text) and is copied unchanged.

Run this after analyze_study.py. Does not call the model or touch logs/.
"""

import csv
import shutil
from pathlib import Path

ANALYSIS_DIR = Path("analysis")
PUBLIC_DIR = Path("public")

ROWS_PUBLIC_FIELDS = [
    "arm",
    "replicate",
    "sample_id",
    "outcome",
    "refusal",
    "parse_failure",
    "declined_insufficient_info",
    "decomposition_fallback",
    "num_subquestions",
]

DETAIL_PUBLIC_FIELDS = [
    "replicate",
    "sample_id",
    "single_arm_outcome_same_item",
    "num_subquestions",
    "any_subanswer_hedged",
]


def strip_csv(src: Path, dst: Path, fields: list[str]) -> int:
    with open(src, newline="", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        rows = [{k: row[k] for k in fields} for row in reader]
    with open(dst, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    PUBLIC_DIR.mkdir(exist_ok=True)

    n = strip_csv(ANALYSIS_DIR / "rows.csv", PUBLIC_DIR / "rows.csv", ROWS_PUBLIC_FIELDS)
    print(f"wrote public/rows.csv ({n} rows, columns: {ROWS_PUBLIC_FIELDS})")

    n = strip_csv(
        ANALYSIS_DIR / "chain_refusal_detail.csv",
        PUBLIC_DIR / "chain_refusal_detail.csv",
        DETAIL_PUBLIC_FIELDS,
    )
    print(f"wrote public/chain_refusal_detail.csv ({n} rows, columns: {DETAIL_PUBLIC_FIELDS})")

    shutil.copyfile(ANALYSIS_DIR / "report.txt", PUBLIC_DIR / "report.txt")
    print("copied analysis/report.txt -> public/report.txt (already aggregate-only)")


if __name__ == "__main__":
    main()
