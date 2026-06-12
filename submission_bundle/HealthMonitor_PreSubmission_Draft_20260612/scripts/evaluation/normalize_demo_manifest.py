"""Normalize public-facing metadata in the frozen demo manifest."""
from __future__ import annotations

import json

from healthmonitor.monitoring_policy import policy_config_dict
from healthmonitor.paths import DEMO_DIR


def main():
    path = DEMO_DIR / "replay_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["manifest_version"] = "competition-demo-v2"
    manifest["generated_with"] = {
        "ensemble": "official three-member logits ensemble",
        "policy": policy_config_dict(),
        "inference_interval_sec": 10,
        "serving_batch_size": 1,
    }
    manifest["display_cases"] = ["119", "100", "201"]
    manifest["boundary_cases"] = ["223", "209"]
    manifest["default_case"] = "119"
    manifest["default_indices"] = {"119": 4, "100": 17, "201": 10}
    manifest["external_support_note"] = (
        "AFib辅助方向由LTAFDB 83记录主分析、84记录敏感性分析支持；"
        "不接入正式报警。"
    )
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    for replay_path in DEMO_DIR.glob("mitdb_*.json"):
        replay = json.loads(replay_path.read_text(encoding="utf-8"))
        replay["record_path"] = f"data/{replay['db']}/{replay['record_id']}"
        replay_path.write_text(json.dumps(replay, ensure_ascii=False, indent=2), encoding="utf-8")
    for meta_path in (DEMO_DIR / "signals").glob("mitdb_*_signal.json"):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["source_record_path"] = f"data/{meta['db']}/{meta['record_id']}"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
