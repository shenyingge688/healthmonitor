"""
download_svdb.py — 下载 MIT-BIH Supraventricular Arrhythmia Database (svdb)
"""
import wfdb
import os

SAVE_DIR = os.path.join("data", "svdb")
os.makedirs(SAVE_DIR, exist_ok=True)

# svdb 包含 78 条 30 分钟记录
records = wfdb.get_record_list("svdb")
print(f"svdb 共 {len(records)} 条记录")

for rec in records:
    try:
        wfdb.dl_database("svdb", SAVE_DIR, records=[rec], annotators=["atr"])
        print(f"  [OK] {rec}")
    except Exception as e:
        print(f"  [FAIL] {rec}: {e}")

print("\nDone.")
