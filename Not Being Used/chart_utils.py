"""Utilities for creating pie charts from a list of values."""

from __future__ import annotations

import re
from collections.abc import Sequence

import matplotlib.pyplot as plt


def get_false_positive_rates_from_log(log_path: str = "LSTM/lstm.log") -> list[float]:
    """
    Return all false positive rate (FPR) values found in an LSTM training log.

    Args:
        log_path: Path to the log file (defaults to ``LSTM/lstm.log``).

    Returns:
        list[float]: FPR values parsed from each epoch row.
    """
    fpr_values: list[float] = []
    epoch_line_pattern = re.compile(r"-\s+INFO\s+-\s+\d+\s+\|")

    with open(log_path, "r", encoding="utf-8") as log_file:
        for line in log_file:
            if not epoch_line_pattern.search(line):
                continue

            columns = [column.strip() for column in line.split("|")]
            if len(columns) < 5:
                continue

            fpr_text = columns[-2]
            try:
                fpr_values.append(float(fpr_text))
            except ValueError:
                continue

    if not fpr_values:
        raise ValueError(f"No FPR values were found in log file: {log_path}")

    return fpr_values


def create_pie_chart(
    values: Sequence[float],
    labels: Sequence[str] | None = None,
    *,
    title: str = "Pie Chart",
    show_percentages: bool = True,
    start_angle: float = 90,
    output_path: str | None = None,
    show: bool = True,
):
    """
    Create and optionally save/show a pie chart from a list of values.

    Args:
        values: Numeric values used to build pie slices.
        labels: Optional labels for each slice. If omitted, labels are auto-generated.
        title: Plot title.
        show_percentages: Whether to display percentages inside slices.
        start_angle: Starting rotation angle for the first slice.
        output_path: Optional output path to save the chart image.
        show: Whether to display the chart window.

    Returns:
        tuple: (figure, axis)
    """
    if not values:
        raise ValueError("values must contain at least one element")

    numeric_values = [float(value) for value in values]

    if any(value < 0 for value in numeric_values):
        raise ValueError("values must be non-negative")

    if sum(numeric_values) == 0:
        raise ValueError("at least one value must be greater than zero")

    if labels is None:
        labels = [f"Slice {index + 1}" for index in range(len(numeric_values))]
    elif len(labels) != len(numeric_values):
        raise ValueError("labels must have the same length as values")

    fig, ax = plt.subplots()
    autopct = "%1.1f%%" if show_percentages else None

    ax.pie(
        numeric_values,
        labels=labels,
        autopct=autopct,
        startangle=start_angle,
    )
    ax.set_title(title)
    ax.axis("equal")

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax


if __name__ == "__main__":
    create_pie_chart([35, 25, 20, 20], ["A", "B", "C", "D"], title="Example Pie Chart")
