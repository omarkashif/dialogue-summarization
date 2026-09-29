"""Create a final-test ROUGE-L length plot from the saved frozen JSON only.

This script intentionally does not load SAMSum, checkpoints, or model code.
It reads the aggregate final-test length analysis produced by the one-time
frozen-protocol evaluation and renders a report-quality comparison plot.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "final_test_evaluation"
    / "frozen_length_stratified_test_results.json"
)
OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "final_test_evaluation"
    / "figures"
    / "rouge_l_by_length_group.png"
)


def main() -> None:
    """Read frozen metrics and write a plot without accessing experimental data."""
    with INPUT_PATH.open(encoding="utf-8") as source:
        payload = json.load(source)

    rows = payload["rows"]
    groups = [row["length_group"].capitalize() for row in rows]
    vanilla = [row["vanilla_rouge_l"] for row in rows]
    attention = [row["attention_rouge_l"] for row in rows]

    positions = list(range(len(groups)))
    width = 0.36
    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.bar(
        [position - width / 2 for position in positions],
        vanilla,
        width,
        label="Vanilla greedy",
    )
    axis.bar(
        [position + width / 2 for position in positions],
        attention,
        width,
        label="Attention greedy",
    )
    axis.set_xticks(positions, groups)
    axis.set_ylim(0, max(vanilla + attention) * 1.15)
    axis.set_xlabel("Training-derived source-dialogue length group")
    axis.set_ylabel("Mean ROUGE-L (F-measure)")
    axis.set_title("Final test ROUGE-L by source-dialogue length")
    axis.grid(axis="y", alpha=0.35)
    axis.legend()
    figure.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=200)
    plt.close(figure)
    print(f"Read frozen aggregate JSON: {INPUT_PATH}")
    print(f"Wrote figure: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
