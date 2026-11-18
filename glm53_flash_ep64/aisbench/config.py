"""EP64 prefix-cache benchmark settings; used in an isolated helper workdir."""

import os
from pathlib import Path

PACKAGE_ROOT = Path(
    os.environ.get("EP64_PACKAGE_ROOT", Path(__file__).resolve().parents[1])
).resolve()
RUN_ROOT = Path(os.environ.get("RUN_ROOT", PACKAGE_ROOT / "runtime" / "aisbench"))
DATASET_PATH = str(RUN_ROOT / "dataset")
WORK_PATH = os.environ.get("WORK_PATH", "/mnt/share/l00887471/benchmark")
MODEL_NAME = "glm"
MODEL_PATH = str(PACKAGE_ROOT / "models" / "glm5.3-flash-10T-ep64-l78-w4a4c8-kda8")
# Run on node0 inside ljt_10t_test. Its API listens on port8000.
HOST_IP = os.environ.get("HOST_IP", "127.0.0.1")
HOST_PORT = os.environ.get("HOST_PORT", "8000")
URL = ""
API_KEY = ""
DEFAULT_PERFORMANCE_TEST = "default_perf"
OUTPUT_DIR = str(RUN_ROOT / "output")
POD_INFO = []

# aisbench_test.py consumes these through CLI arguments, not config.py.
# run_aisbench.sh below explicitly passes each value to that CLI.
INPUT_LEN = int(os.environ.get("INPUT_LEN", "262132"))
OUTPUT_LEN = int(os.environ.get("OUTPUT_LEN", "4096"))
DATA_NUM = int(os.environ.get("DATA_NUM", "256"))
CONCURRENCY = int(os.environ.get("CONCURRENCY", "256"))
DP = 64
NPU_NUM = 64
REQUEST_RATE = 0
SEED = 1024
