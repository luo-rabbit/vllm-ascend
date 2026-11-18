#!/usr/bin/env bash
set -eo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PACKAGE_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
# shellcheck source=../scripts/cluster_config.sh
source "$PACKAGE_ROOT/scripts/cluster_config.sh"
DRY_RUN=0
if [[ "${1:-}" == --dry-run ]]; then
  DRY_RUN=1
  shift
fi
if [[ "$#" != 0 ]]; then
  echo 'Usage: run_aisbench.sh [--dry-run]; override counts with environment variables' >&2
  exit 2
fi
if [[ "$DRY_RUN" == 0 ]]; then
  source "$VENV_PATH/bin/activate"
fi
export EP64_PACKAGE_ROOT="$PACKAGE_ROOT"
export WORK_PATH
export MAX_MODEL_LEN
export HOST_IP=${HOST_IP:-127.0.0.1}
export HOST_PORT=${HOST_PORT:-$API_PORT}
export EXPECTED_VENV="$VENV_PATH"
export AISBENCH_TOOLS
export BENCH_DRY_RUN="$DRY_RUN"
export PYTHONUNBUFFERED=1
python3 -u - "$SCRIPT_DIR/config.py" <<'PY'
import importlib.util
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile

c = runpy.run_path(sys.argv[1])
args = [
    "--input_len", str(c["INPUT_LEN"]), "--output_len", str(c["OUTPUT_LEN"]),
    "--data_num", str(c["DATA_NUM"]), "--concurrency", str(c["CONCURRENCY"]),
    "--request_rate", str(c["REQUEST_RATE"]), "--dataset_type", "prefix_cache",
    "--repeat_rate", "1", "--prefix_test", "--seed", str(c["SEED"]), "--dp", str(c["DP"]),
    "--npu_num", str(c["NPU_NUM"]),
]
print(json.dumps({"argv": ["python3", "aisbench_test.py", *args],
                  "WORK_PATH": c["WORK_PATH"], "MODEL_PATH": c["MODEL_PATH"],
                  "HOST_IP": c["HOST_IP"], "HOST_PORT": c["HOST_PORT"],
                  "DATASET_PATH": c["DATASET_PATH"], "OUTPUT_DIR": c["OUTPUT_DIR"]}, indent=2))
if os.environ["BENCH_DRY_RUN"] == "1":
    sys.exit(0)
assert c["DATA_NUM"] >= c["CONCURRENCY"], "data_num must cover requested concurrency"
assert c["INPUT_LEN"] + c["OUTPUT_LEN"] <= int(os.environ.get("MAX_MODEL_LEN", "268288"))
assert Path(sys.prefix).resolve() == Path(os.environ["EXPECTED_VENV"]).resolve()
root = Path(c["WORK_PATH"]).resolve()
spec = importlib.util.find_spec("ais_bench")
assert spec and spec.origin and Path(spec.origin).resolve().is_relative_to(root), spec
binary = shutil.which("ais_bench")
assert binary and Path(binary).parent.resolve() == Path(sys.prefix).resolve() / "bin", binary
assert c["MODEL_NAME"] == "glm" and not c["URL"]
assert (Path(c["MODEL_PATH"]) / "tokenizer.json").is_file()
tools = Path(os.environ["AISBENCH_TOOLS"])
files = ["aisbench_test.py", "generate_dataset.py", "data_picker.py", "save_file.py",
         "cal_prefix_hit_rate.py", "default_api.py", "GSM8K.jsonl"]
for name in files:
    assert (tools / name).is_file(), str(tools / name)
run_root = Path(c["RUN_ROOT"])
run_root.mkdir(parents=True, exist_ok=True)
work = Path(tempfile.mkdtemp(prefix="helpers_", dir=run_root))
for name in files:
    shutil.copy2(tools / name, work / name)
shutil.copy2(sys.argv[1], work / "config.py")
print("Isolated helper workdir:", work)
print("AISBench source:", spec.origin)
result = subprocess.run([sys.executable, "-u", "aisbench_test.py", *args], cwd=work, check=False)
(work / "wrapper_exit_code.txt").write_text(str(result.returncode) + "\n")
print("Check AISBench final request success/failure counts; wrapper exit code alone is insufficient.")
sys.exit(result.returncode)
PY
