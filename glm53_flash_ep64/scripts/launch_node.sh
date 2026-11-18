#!/usr/bin/env bash
set -eo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PACKAGE_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
# shellcheck source=cluster_config.sh
source "${SCRIPT_DIR}/cluster_config.sh"
NODE_INDEX=${1:?Usage: launch_node.sh 0..7 [--dry-run] [--no-profile]}
shift
case "$NODE_INDEX" in [0-7]) ;; *) echo 'Node index must be 0..7' >&2; exit 2 ;; esac
DRY_RUN=0
PROFILE=1
for option in "$@"; do
  case "$option" in
    --dry-run) DRY_RUN=1 ;;
    --no-profile) PROFILE=0 ;;
    *) echo "Unknown option: $option" >&2; exit 2 ;;
  esac
done
if [[ ${#NODE_IPS[@]} -ne 8 ]]; then
  echo 'Configure exactly eight node IPs in cluster_config.sh' >&2
  exit 2
fi
if [[ "$DRY_RUN" == 0 ]]; then
  for node_ip in "${NODE_IPS[@]}"; do
    case "$node_ip" in *ip*|'') echo 'Replace all eight IP placeholders first' >&2; exit 2 ;; esac
  done
  # The default experiment uses the image-provided CANN, not an inherited
  # isolated PoolKey deployment. FLA itself remains a required dependency.
  case "${ASCEND_CUSTOM_OPP_PATH-}:${LD_LIBRARY_PATH-}" in
    */custom_ops/poolkey/*) echo 'Use a clean shell for image-provided CANN' >&2; exit 2 ;;
  esac
  source "$CANN_ENV"
  source "$ATB_ENV"
  source "$VENV_PATH/bin/activate"
fi
set -u
case "$SHARED_EXPERT_MULTISTREAM:$FORCE_EPLB" in
  true:true|true:false|false:true|false:false) ;;
  *) echo 'Shared-expert and EPLB flags must be true or false' >&2; exit 2 ;;
esac
if [[ "$SPEC_TOKENS" != 7 ]]; then
  echo 'This validated KDA configuration uses DSpark7 (8 target verification tokens)' >&2
  exit 2
fi
MODEL="$PACKAGE_ROOT/models/glm5.3-flash-10T-ep64-l78-w4a4c8-kda8"
DRAFT="$PACKAGE_ROOT/models/draft-glm53-dspark-swa-dummy-spec7"
test -s "$MODEL/config.json"
test -s "$MODEL/quant_model_description.json"
test -s "$DRAFT/config.json"
LOCAL_IP=${NODE_IPS[$NODE_INDEX]}
NODE0_IP=${NODE_IPS[0]}
DP_START=$((NODE_INDEX * 8))
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
RUN_ROOT=${RUN_ROOT:-$PACKAGE_ROOT/runtime/$RUN_ID}
PROF_DIR="$RUN_ROOT/prof_data/node$NODE_INDEX"
export HCCL_IF_IP="$LOCAL_IP"
export GLOO_SOCKET_IFNAME="$NIC_NAME"
export TP_SOCKET_IFNAME="$NIC_NAME"
export HCCL_SOCKET_IFNAME="$NIC_NAME"
export VLLM_HOST_IP="$LOCAL_IP"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_USE_V2_MODEL_RUNNER=1
export HCCL_BUFFSIZE
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
export HCCL_EXEC_TIMEOUT=3600
export HCCL_CONNECT_TIMEOUT=1200
export VLLM_ENGINE_READY_TIMEOUT_S=1800
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
SPEC_CONFIG="{\"method\":\"dspark\",\"model\":\"$DRAFT\",\"num_speculative_tokens\":7,\"draft_tensor_parallel_size\":1,\"draft_sample_method\":\"greedy\",\"enforce_eager\":true,\"kv_cache_dtype\":\"bfloat16\",\"draft_load_config\":{\"load_format\":\"dummy\"}}"
ADDITIONAL_CONFIG="{\"enable_fused_mc2\":0,\"enable_force_eplb\":$FORCE_EPLB,\"multistream_overlap_shared_expert\":$SHARED_EXPERT_MULTISTREAM}"
CMD=(vllm serve "$MODEL"
  --host 0.0.0.0 --port "$API_PORT"
  --data-parallel-size 64 --data-parallel-size-local 8
  --data-parallel-start-rank "$DP_START"
  --data-parallel-address "$NODE0_IP" --data-parallel-rpc-port "$RPC_PORT"
  --tensor-parallel-size 1 --enable-expert-parallel
  --seed 1024 --served-model-name glm --load-format dummy
  --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --trust-remote-code --quantization ascend
  --kv-cache-dtype fp8 --attention_config.indexer_kv_dtype fp8
  --gpu-memory-utilization "$GPU_MEM_UTIL" --async-scheduling
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
  --enable-prefix-caching --limit-mm-per-prompt '{"image":0,"video":0}'
  --speculative-config "$SPEC_CONFIG" --additional-config "$ADDITIONAL_CONFIG"
)
if [[ "$NODE_INDEX" == 0 ]]; then
  CMD+=(--api-server-count 1)
else
  CMD+=(--headless)
fi
if [[ "$PROFILE" == 1 ]]; then
  CMD+=(--profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$PROF_DIR\",\"torch_profiler_with_stack\":false,\"torch_profiler_record_shapes\":true,\"ignore_frontend\":true}")
fi
printf 'node=%s dp_start=%s local_ip=%s node0_ip=%s\n' "$NODE_INDEX" "$DP_START" "$LOCAL_IP" "$NODE0_IP"
printf 'command:'
printf ' %q' "${CMD[@]}"
printf '\n'
if [[ "$DRY_RUN" == 1 ]]; then
  exit 0
fi
mkdir -p "$RUN_ROOT/server_logs" "$PROF_DIR"
cd "$PACKAGE_ROOT"
"${CMD[@]}" 2>&1 | tee "$RUN_ROOT/server_logs/node${NODE_INDEX}.log"
