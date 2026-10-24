"""Aggregate and plot completed behavior diagnostics without training or simulation."""

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import fmean


FULL_WINDOW_MEANS = {
    "mean_duration_s": "duration_s",
    "mean_episode_return": "episode_return",
    "mean_linear_xy_rmse_mps": "linear_xy_rmse_mps",
    "mean_yaw_rmse_radps": "yaw_rmse_radps",
    "mean_tilt_mean_deg": "tilt_mean_deg",
    "mean_tilt_max_deg": "tilt_max_deg",
    "mean_action_std": "mean_action_std",
}
COMMON_WINDOW_MEANS = {
    "mean_common_window_duration_s": "duration_s",
    "mean_common_window_linear_xy_rmse_mps": "linear_xy_rmse_mps",
    "mean_common_window_yaw_rmse_radps": "yaw_rmse_radps",
    "mean_common_window_tilt_mean_deg": "tilt_mean_deg",
}


def _key(row):
    return int(row["iteration"]), row["case"], int(row["seed"])


def _finite_mean(rows, name):
    values = [float(row[name]) for row in rows]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError(f"{name} must contain nonempty finite values")
    return fmean(values)


def _aggregate(rows, common_rows, *, iteration, task, case=None):
    """Equal weight per recorded episode; fall times exclude non-fall cutoffs."""
    trials = len(rows)
    fallen = [row for row in rows if row["fall_detected"]]
    if any(row["fall_time_s"] is None for row in fallen):
        raise ValueError("a detected fall must have a fall_time_s")
    if any(row["fall_time_s"] is not None for row in rows if not row["fall_detected"]):
        raise ValueError("non-fall episodes must keep fall_time_s null")
    result = {
        "iteration": iteration,
        "task": task,
        "trials": trials,
        "falls": len(fallen),
        "fall_fraction": len(fallen) / trials,
        "cutoffs": sum(bool(row["evaluation_cutoff"]) for row in rows),
        "terminated": sum(bool(row["terminated"]) for row in rows),
        "truncated": sum(bool(row["truncated"]) for row in rows),
        "mean_fall_time_s": _finite_mean(fallen, "fall_time_s") if fallen else None,
        "common_window_trials": len(common_rows),
        "mean_common_window_steps": _finite_mean(common_rows, "steps"),
    }
    if case is not None:
        result["case"] = case
    for target, source in FULL_WINDOW_MEANS.items():
        result[target] = _finite_mean(rows, source)
    for target, source in COMMON_WINDOW_MEANS.items():
        result[target] = _finite_mean(common_rows, source)
    for field in ("mean_linear_velocity_b", "mean_angular_velocity_b"):
        if any(len(row[field]) != 3 for row in rows):
            raise ValueError(f"{field} must have three body-frame components")
        result[field] = [fmean(float(row[field][axis]) for row in rows) for axis in range(3)]
        if not all(math.isfinite(value) for value in result[field]):
            raise ValueError(f"{field} must be finite")
    return result


def build_aggregates(summary):
    """Match each common prefix to its episode and group by case and task."""
    episodes = summary["episodes"]
    windows = summary["common_windows"]
    if not episodes or not windows:
        raise ValueError("episodes and common_windows must be nonempty")
    episode_keys = [_key(row) for row in episodes]
    window_keys = [_key(row) for row in windows]
    if len(set(episode_keys)) != len(episode_keys) or len(set(window_keys)) != len(window_keys):
        raise ValueError("duplicate checkpoint/case/seed diagnostic rows")
    if set(episode_keys) != set(window_keys):
        raise ValueError("every episode must have exactly one matched common window")
    common_by_key = dict(zip(window_keys, windows))
    case_tasks = {}
    for row in episodes:
        previous = case_tasks.setdefault(row["case"], row["task"])
        if previous != row["task"]:
            raise ValueError("a case cannot belong to multiple tasks")
        common = common_by_key[_key(row)]
        if not 0 < int(common["steps"]) <= int(row["steps"]):
            raise ValueError("common prefix steps must be within the recorded episode")
        if not 0 < float(common["duration_s"]) <= float(row["duration_s"]) + 1e-9:
            raise ValueError("common prefix duration must be within the recorded episode")
    # A common window is the shortest duration among the paired checkpoints,
    # so all checkpoint rows for each case/seed must use the same prefix length.
    prefixes = {}
    for row in windows:
        pair = row["case"], int(row["seed"])
        current = int(row["steps"]), float(row["duration_s"])
        if prefixes.setdefault(pair, current) != current:
            raise ValueError("checkpoint comparisons must use equal case/seed common prefixes")
    case_aggregates, task_aggregates = [], []
    for group_name, groups, output in (
            ("case", list(case_tasks), case_aggregates),
            ("task", list(dict.fromkeys(case_tasks.values())), task_aggregates)):
        for iteration in sorted({int(row["iteration"]) for row in episodes}):
            for group in groups:
                selected = [row for row in episodes
                            if int(row["iteration"]) == iteration and row[group_name] == group]
                if not selected:
                    raise ValueError("each checkpoint must have every diagnostic case and task")
                common = [common_by_key[_key(row)] for row in selected]
                task = case_tasks[group] if group_name == "case" else group
                output.append(_aggregate(selected, common, iteration=iteration, task=task,
                                         case=group if group_name == "case" else None))
    return task_aggregates, case_aggregates


def write_metrics(path, case_aggregates):
    """One row per case/checkpoint, retaining explicit units in column names."""
    rows = []
    for source in case_aggregates:
        row = dict(source)
        for name in ("mean_linear_velocity_b", "mean_angular_velocity_b"):
            vector = row.pop(name)
            for axis, value in zip(("x", "y", "z"), vector):
                row[f"{name}_{axis}"] = value
        rows.append(row)
    fields = ["iteration", "task", "case"] + [key for key in rows[0]
                                                if key not in {"iteration", "task", "case"}]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_comparison(path, summary, case_aggregates):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    cases = list(dict.fromkeys(row["case"] for row in summary["episodes"]))
    iterations = sorted({int(row["iteration"]) for row in case_aggregates})
    values = {(row["iteration"], row["case"]): row for row in case_aggregates}
    panels = (
        ("mean_duration_s", "Recorded episode duration", "Seconds"),
        ("fall_fraction", "Episodes ending in a detected fall", "Fraction of trials"),
        ("mean_common_window_linear_xy_rmse_mps", "XY velocity error: matched common prefix", "RMSE (m/s)"),
        ("mean_common_window_yaw_rmse_radps", "Yaw rate error: matched common prefix", "RMSE (rad/s)"),
        ("mean_common_window_tilt_mean_deg", "Body tilt: matched common prefix", "Mean tilt (degrees)"),
        ("mean_tilt_max_deg", "Peak body tilt over each full episode", "Mean of episode maxima (degrees)"),
    )
    x = np.arange(len(cases))
    width = .8 / len(iterations)
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    for ax, (field, title, unit) in zip(axes.flat, panels):
        for index, iteration in enumerate(iterations):
            offset = (index - (len(iterations) - 1) / 2) * width
            ax.bar(x + offset, [values[iteration, case][field] for case in cases],
                   width=width, label=f"Iteration {iteration}")
        ax.set_title(title)
        ax.set_ylabel(unit)
        ax.set_xticks(x, [case.replace("_", " ").replace("-", " ").title() for case in cases],
                      rotation=22, ha="right")
        ax.set_ylim(bottom=0)
        if field == "fall_fraction":
            ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=.25)
        ax.set_axisbelow(True)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .965), ncol=len(iterations))
    fig.suptitle("Fixed-command behavior diagnostics: frozen normalization, actor mean actions", fontsize=16, y=.995)
    cap = summary["metadata"]["max_seconds"]
    fig.text(.5, .025,
             f"Recording stops at a fall, environment termination/timeout, or the {cap:g} s evaluation cap. "
             "Reaching the cap is not counted as success.\n"
             "Error and mean-tilt panels use the same prefix length across checkpoints for each case and seed; "
             "peak tilt uses each complete recorded window.",
             ha="center", va="bottom", fontsize=10)
    fig.tight_layout(rect=(0, .095, 1, .935))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    from training.paths import DIAGNOSTICS_DIR
    parser.add_argument("--run", type=Path, default=DIAGNOSTICS_DIR,
                        help="Diagnostic output directory (or its summary.json)")
    args = parser.parse_args()
    source = args.run if args.run.is_file() else args.run / "summary.json"
    if not source.is_file():
        parser.error(f"diagnostic summary does not exist: {source}")
    root = source.parent
    summary = json.loads(source.read_text(encoding="utf-8"))
    aggregates, case_aggregates = build_aggregates(summary)
    summary["aggregates"] = aggregates
    summary["case_aggregates"] = case_aggregates
    write_metrics(root / "metrics.csv", case_aggregates)
    plot_comparison(root / "comparison.png", summary, case_aggregates)
    source.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"task_groups": len(aggregates), "case_groups": len(case_aggregates),
                      "summary": str(source), "metrics": str(root / "metrics.csv"),
                      "comparison": str(root / "comparison.png")}, indent=2))


if __name__ == "__main__":
    main()
