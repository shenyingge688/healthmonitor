"""Central path resolution for the HealthMonitor project."""
from __future__ import annotations

import os
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(
    os.environ.get("HEALTHMONITOR_ROOT", PACKAGE_DIR.parent)
).expanduser().resolve()


def _path_from_env(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, default)).expanduser().resolve()


DATA_DIR = _path_from_env("HEALTHMONITOR_DATA_DIR", PROJECT_ROOT / "data")
TRAJECTORY_DATASET_DIR = _path_from_env(
    "HEALTHMONITOR_TRAJECTORY_DIR",
    PROJECT_ROOT / "datasets" / "trajectory_official",
)
ARTIFACTS_DIR = _path_from_env("HEALTHMONITOR_ARTIFACTS_DIR", PROJECT_ROOT / "artifacts")
MODELS_DIR = _path_from_env("HEALTHMONITOR_MODELS_DIR", PROJECT_ROOT / "models")
OFFICIAL_ENSEMBLE_DIR = _path_from_env(
    "HEALTHMONITOR_ENSEMBLE_DIR",
    MODELS_DIR / "official_ensemble",
)
PRETRAINED_MODELS_DIR = _path_from_env(
    "HEALTHMONITOR_PRETRAINED_DIR",
    MODELS_DIR / "pretrained",
)
DEMO_DIR = _path_from_env("HEALTHMONITOR_DEMO_DIR", ARTIFACTS_DIR / "demo")
EVIDENCE_DIR = _path_from_env("HEALTHMONITOR_EVIDENCE_DIR", ARTIFACTS_DIR / "evidence")
POLICY_DIR = _path_from_env("HEALTHMONITOR_POLICY_DIR", ARTIFACTS_DIR / "policies")
FIGURES_DIR = _path_from_env("HEALTHMONITOR_FIGURES_DIR", ARTIFACTS_DIR / "figures")
EXPLAINABILITY_DIR = _path_from_env(
    "HEALTHMONITOR_EXPLAINABILITY_DIR",
    ARTIFACTS_DIR / "explainability",
)
DOCS_DIR = _path_from_env("HEALTHMONITOR_DOCS_DIR", PROJECT_ROOT / "docs")
