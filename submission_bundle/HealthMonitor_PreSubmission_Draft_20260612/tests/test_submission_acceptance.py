"""Pre-submission acceptance checks for public artifacts."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _docx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as zf:
        chunks = []
        for name in zf.namelist():
            if name.startswith("word/") and name.endswith(".xml"):
                chunks.append(zf.read(name).decode("utf-8", errors="ignore"))
        return "\n".join(chunks)


def main_test():
    banned = ["Round", "current_core_v7", "V7", "V8"]
    public_paths = [
        ROOT / "artifacts" / "metric_registry.json",
        ROOT / "artifacts" / "demo" / "replay_manifest.json",
        ROOT / "README.md",
        ROOT / "PROJECT_FILE_MAP.md",
    ]
    for path in public_paths:
        text = path.read_text(encoding="utf-8")
        hits = [term for term in banned if term in text]
        if hits:
            raise AssertionError(f"{path} contains banned terms: {hits}")

    for docx in (ROOT / "docs").glob("*.docx"):
        text = _docx_text(docx)
        hits = [term for term in banned + ["C:\\HealthMonitor"] if term in text]
        if hits:
            raise AssertionError(f"{docx} contains banned terms: {hits}")

    manifest = json.loads(
        (ROOT / "artifacts" / "demo" / "replay_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["manifest_version"] == "competition-demo-v2"
    assert manifest["display_cases"] == ["119", "100", "201"]
    assert manifest["boundary_cases"] == ["223", "209"]

    registry = json.loads(
        (ROOT / "artifacts" / "metric_registry.json").read_text(encoding="utf-8")
    )
    assert registry["policy_config_version"] == "overall-risk-v1"
    cases = registry["demo_cases"]
    assert cases["pvc_alarm"]["display_index"] == 5
    assert cases["pvc_alarm"]["future_vote_count"] == 3
    assert cases["pvc_alarm"]["agreement_level"] == "high"
    assert cases["pvc_alarm"]["official_alert_active"] is True
    assert cases["normal_no_alarm"]["display_index"] == 18
    assert cases["normal_no_alarm"]["official_alert_active"] is False
    assert cases["afib_auxiliary"]["display_index"] == 11
    assert cases["afib_auxiliary"]["agreement_level"] == "high"
    assert cases["afib_auxiliary"]["signal_quality_score"] == 1.0
    print("Submission acceptance checks passed.")


if __name__ == "__main__":
    main_test()
