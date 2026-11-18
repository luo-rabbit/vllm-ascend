#!/usr/bin/env bash
# Edit the eight data0.3001 addresses here, in node0..node7 order.
# If each host has a separate disk, clone/copy this bundle to every host.
NODE_IPS=(
  "机1ip" "机2ip" "机3ip" "机4ip"
  "机5ip" "机6ip" "机7ip" "机8ip"
)
NIC_NAME=${NIC_NAME:-data0.3001}
VENV_PATH=${VENV_PATH:-/mnt/share/l00887471/10t/.venv-312}
CANN_ENV=${CANN_ENV:-/usr/local/Ascend/cann/set_env.sh}
ATB_ENV=${ATB_ENV:-/usr/local/Ascend/nnal/atb/set_env.sh}
WORK_PATH=${WORK_PATH:-/mnt/share/l00887471/benchmark}
AISBENCH_TOOLS=${AISBENCH_TOOLS:-/mnt/share/l00887471/aisbench_auto_tools_prefix}
RPC_PORT=${RPC_PORT:-12321}
API_PORT=${API_PORT:-8000}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-268288}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-64}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-512}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.93}
HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-1024}
SPEC_TOKENS=${SPEC_TOKENS:-7}
SHARED_EXPERT_MULTISTREAM=${SHARED_EXPERT_MULTISTREAM:-true}
FORCE_EPLB=${FORCE_EPLB:-true}
