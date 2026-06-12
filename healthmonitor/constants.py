"""
constants.py — 项目中所有共享常量定义
"""
from .paths import ARTIFACTS_DIR, DATA_DIR as ROOT_DATA_DIR, MODELS_DIR, PROJECT_ROOT
from .paths import TRAJECTORY_DATASET_DIR

# =========================================================
# 信号/采样参数
# =========================================================
TARGET_FS = 250
HISTORY_SEC = 600          # 输入: 过去 10 分钟
PREDICT_SEC = 300          # 预测窗口: 未来 5 分钟
WINDOW_SEC = 30            # 单窗口时长
OVERLAP_STRIDE_SEC = 15    # 窗口步长
PTS_PER_WIN = WINDOW_SEC * TARGET_FS            # 7500
STRIDE_PTS = OVERLAP_STRIDE_SEC * TARGET_FS     # 3750
N_WINDOWS = (HISTORY_SEC - WINDOW_SEC) // OVERLAP_STRIDE_SEC + 1  # 39
HISTORY_PTS = (N_WINDOWS - 1) * STRIDE_PTS + PTS_PER_WIN         # 150000
API_BUFFER_SIZE = 180000

# =========================================================
# 分类体系
# =========================================================
CLASS_NAMES = [
    "正常窦性心律",
    "室性早搏 (PVC)",
    "心房颤动 (AFib)",
    "心室颤动 (VF)",
    "室性心动过速 (VT)",
    "房速/室上速 (AT/SVT)",
]

CLASS_COLORS = [
    "#10B981", "#F59E0B", "#F97316",
    "#991B1B", "#EF4444", "#EAB308",
]

INFERENCE_MAP = {
    0: {"title": "未见明显节律异常", "color": "#10B981"},
    1: {"title": "室性期前收缩 (PVC) 风险", "color": "#F59E0B"},
    2: {"title": "心房颤动 (AFib) 风险", "color": "#F97316"},
    3: {"title": "心室颤动 (VF) 风险", "color": "#991B1B"},
    4: {"title": "室性心动过速 (VT) 风险", "color": "#EF4444"},
    5: {"title": "房速/室上速 (AT/SVT) 风险", "color": "#EAB308"},
}

# 优先级映射: VF(3) > VT(4) > AFIB(2) > AT/SVT(5) > PVC(1) > Normal(0)
CLASS_PRIORITY = {3: 6, 4: 5, 2: 4, 5: 3, 1: 2, 0: 1}

# =========================================================
# 路径
# =========================================================
BASE_DIR = str(PROJECT_ROOT)
DATA_DIR = str(ROOT_DATA_DIR)
SAVE_DIR = str(TRAJECTORY_DATASET_DIR)
MODEL_DIR = str(MODELS_DIR)
RESULTS_DIR = str(ARTIFACTS_DIR)
