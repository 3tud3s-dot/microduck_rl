"""Canonical retained experiment and checkpoint locations."""

from pathlib import Path

TRAINING_ROOT = Path(__file__).resolve().parent
RUNS_ROOT = TRAINING_ROOT / "runs"
REFERENCE_RUN = RUNS_ROOT / "medium_trial"
INITIAL_CHECKPOINT = REFERENCE_RUN / "initial.pt"
LATEST_CHECKPOINT = REFERENCE_RUN / "train" / "iteration_0100.pt"
DIAGNOSTICS_DIR = REFERENCE_RUN / "diagnostics"
