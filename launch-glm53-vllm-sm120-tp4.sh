#!/usr/bin/env bash

set -euo pipefail

# GLM-5.3-Flash (native FP8 checkpoint, zai-org) on a single x86_64 node with
# 4x RTX PRO 6000 Blackwell (SM120, 96 GB each), vLLM TP4.
#
# Differences vs the DGX Spark TP2/TP4 launchers:
#  - single node: no IB fabric env, no --nnodes/--master-addr, no NFS
#  - weights: the FULL FP8 checkpoint (328 GB -> ~82 GB/rank) served straight
#    from the HF cache snapshot; fp8 KV cache is mandatory for headroom
#  - PCIe P2P ON by default: `nvidia-smi topo -p2p r` shows OK across all
#    Pro 6000s and it benched clean+fast here (the glm-5.2-sm120 recipe's
#    "P2P must be off" claim did NOT reproduce on this box)
#  - arch list 12.0a (sm_120a), not 12.1a
#  - flagship defaults (validated 2026-08-27): CUDA graphs + P2P + MTP-4 +
#    262K context -> ~172 tok/s decode, TTFT 0.059s. fp8 KV pool pinned with
#    --kv-cache-memory=10934056960 (vLLM's own "fully utilize" suggestion):
#    1,428,980 tokens, 5.45x full 262K contexts (was 557,795 / 2.13x unpinned).
#    Ladder: eager/no-P2P 10.6 tok/s -> graphs+P2P ~105 -> +MTP-4 ~172.
#  - persistent compile caches on /cache (host /var/tmp/glm53-sm120-vllm-cache):
#    vLLM (deep_gemm compile, flashinfer-autotune), FlashInfer JIT, TileLang,
#    Triton and TorchInductor artifacts survive restarts, so the multi-minute
#    TileLang/inductor JIT compiles from the 2026-08-27 startup log are paid
#    once, not per boot.

IMAGE="${IMAGE:-glm53:sm120-v1}"
NAME="vllm_glm53_sm120"
MODEL="zai-org/GLM-5.3-Flash"
CACHE_HOST_PATH="/var/tmp/glm53-sm120-vllm-cache"  # HF_HOME + all JIT/compile caches (vllm, flashinfer, tilelang, triton, inductor)
PORT="${PORT:-11110}"
GPUS="${GPUS:-0,1,2,3}"          # 4x RTX PRO 6000 (idle; 5,6 are 5090s, 4,7 spare Pros)
MAX_LEN="${MAX_LEN:-262144}"     # pool 1,428,980 tokens w/ MTP-4 -> 5.45x full-context concurrency
GMU="${GMU:-0.95}"
P2P_DISABLE="${P2P_DISABLE:-0}"  # PCIe P2P verified OK across the Pro 6000s (huge win vs =1)
EAGER="${EAGER-}"  # CUDA graphs by default (10x decode); set EAGER=--enforce-eager to disable
MTP_DEFAULT='--speculative-config {"method":"mtp","num_speculative_tokens":4}'
EXTRA_ARGS="${EXTRA_ARGS-$MTP_DEFAULT}"  # MTP-4 on by default; EXTRA_ARGS="" to disable

#test -f "$SNAPSHOT/config.json"
mkdir -p "$CACHE_HOST_PATH"
docker rm -f "$NAME" 2>/dev/null || true

docker run -d \
  --name "$NAME" --restart no \
  --gpus "\"device=$GPUS\"" \
  --ipc host --shm-size 64g \
  --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
  -p "$PORT:$PORT" \
  -v "$CACHE_HOST_PATH:/cache" \
  -v "$HOME/.cache/huggingface:/cache/huggingface:ro" \
  -e HF_HOME=/cache/huggingface \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e TORCH_CUDA_ARCH_LIST=12.0 -e FLASHINFER_CUDA_ARCH_LIST=12.0a \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e NCCL_P2P_DISABLE=$P2P_DISABLE -e NCCL_NVLS_ENABLE=0 -e NCCL_CUMEM_ENABLE=0 \
  -e NCCL_DEBUG=WARN -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  "$IMAGE" \
    "$MODEL" \
    --kv-cache-memory=10934056960 \
    --served-model-name glm-5.3-flash \
    --host 0.0.0.0 --port "$PORT" \
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization "$GMU" \
    --max-model-len "$MAX_LEN" \
    --max-num-seqs 6 --block-size 2304 \
    --kv-cache-dtype fp8_e4m3 \
    --tool-call-parser glm47 --enable-auto-tool-choice \
    --reasoning-parser glm45 \
    --distributed-executor-backend mp \
    --load-format fastsafetensors \
    --max-num-batched-tokens 8192 \
    --async-scheduling \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --enable-prompt-tokens-details \
    --default-chat-template-kwargs '{"reasoning_effort":"high"}' \
    $EAGER $EXTRA_ARGS

echo "launched $NAME gpus=$GPUS tp=4 max_len=$MAX_LEN gmu=$GMU p2p_disable=$P2P_DISABLE eager=[$EAGER]"
sleep 2
docker ps --format '{{.Names}} {{.Status}}' | grep "$NAME" || {
  echo "$NAME exited; inspect with: docker logs $NAME" >&2
  exit 1
}
