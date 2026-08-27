# GLM-5.3-Flash on SM120 — 4x RTX PRO 6000 Blackwell, single node (epyc)

Deployment date: 2026-08-27 (day 1 after model release)
Endpoint: `http://192.168.50.194:8010/v1`
Model: `glm-5.3-flash` — **native FP8 checkpoint** (zai-org, 328 GB), NOT the NVFP4 quant
Final status: **SERVING, 262,144-token context, ~172 tok/s decode with MTP-4,
1,428,980-token fp8 KV pool (5.45x @262K), all validation gates green**

This is the sm120 port of the sm121 (DGX Spark) deployment documented in
[DEPLOY-REPORT.md](DEPLOY-REPORT.md). Every sm121 bug was re-verified on sm120
silicon (RTX PRO 6000 Blackwell Workstation, cap 12.0) with the repo's probe
scripts BEFORE building — no patch went in on faith.

## 1. Hardware and lane differences vs the Sparks

| | 2x DGX Spark (sm121) | epyc (sm120) |
|---|---|---|
| GPUs | 2x GB10, 120 GB unified | 4x RTX PRO 6000 Blackwell, 96 GB GDDR7 each (of 6; +2 5090s untouched) |
| Interconnect | IB fabric, 2-node NCCL | single node, PCIe P2P (verified OK across Pro 6000s) |
| Weights | LibertAIDAI NVFP4 (182 GiB) over NFS | **zai-org native FP8 (328 GB)** from local HF cache — 76.4 GiB/rank at TP4 |
| Image | `radixark/vllm-glm53-flash:sm121-v9` (arm64) | `glm53:sm120-v1` (amd64, `docker/Dockerfile.glm53-sm120`) |
| Arch list | `12.1a` | `12.0` / `12.0a` |
| Memory model | unified — phantom KV allocations, dmesg OOM kills | discrete — vLLM profiling is truthful, no flush ritual |

The FP8 checkpoint was already fully local (all 62 shards); TP4 shards it to
76.4 GiB/rank with ~11 GiB headroom — so sm120 serves the *unquantized* native
weights, a quality tier above the Spark deployment's NVFP4.

## 2. Bug-by-bug: what sm121 taught, what sm120 confirmed

Probes run in the stock day-0 amd64 image (`vllm/vllm-openai:glm53-flash`,
same vLLM commit `0.1.dev20051+g487ecf187` as the arm64 day-0 image):

| # | sm121 bug | sm120 result | Fix carried over |
|---|---|---|---|
| 1 | NoPE MLA vs packed `fp8_ds_mla` SM120 backend (pe_dim=64 assert) | identical code path, identical death | v1 patch: SM90 NoPE backend → `major in (9,12)`, FA2 off-Hopper |
| 2 | FlashInfer 0.6.17 FA2 MLA NaN on 64–256-row batches | **worse: NaN on EVERY shape incl. 1-row decode** (`probe_fa2_bisect.py` 7/7 NaN) | 0.6.18 nightly → 7/7 clean |
| 3 | nightly downgrades NCCL→2.29.7, skews cutlass-dsl 4.7.0/4.6.2 | **reproduces identically on x86** (pip audit) | re-pin 2.30.7 + 4.6.2 |
| 4 | PDL race on KDA state kernels | same gate (`major >= 9`), same kernels | gated to `(9,10)` |
| 5 | indexer uninit top-k (`torch.empty`) | arch-independent | `patch_v7.py` unchanged |
| 6 | fp8 KV smem over-request in `mla.cuh` | **patch works on sm120: all shapes clean, rel_err 0.0044** (`probe_fa2_fp8.py`) | `patch_v8_fp8.py` unchanged |
| 7 | `--block-size 2304` (DeepGEMM arch-12 pool-page rule) | DeepGEMM keys on `major==12` | serve flag kept |

Probe subtlety: in a single process, a failed `fa3` JIT attempt poisons a
subsequent explicit-`fa2` wrapper (`no kernel image`); fresh-process explicit
fa2 is clean at both H=32 (TP2) and H=16 (TP4) geometry. vLLM never attempts
fa3 on cap-12, so the patched backend is unaffected.

FlashInfer 0.6.18's auto-selection now warns that fa2 is "not Blackwell-native"
and points at trtllm-gen / `backend="cutlass"` for MLA decode on SM120 —
untested here, a candidate future perf lever.

## 3. What sm120 did NOT need

- No IB/NCCL fabric env, no 2-node rendezvous choreography, no worker-first ordering.
- No NFS: weights served from the local HF cache snapshot (25.9 s weight load vs 610 s).
- No cache-flush ritual, no GB10-style edge-riding: discrete VRAM profiling is
  honest. gmu 0.92 just works. (Later the same day we did pin
  `--kv-cache-memory=10934056960` — vLLM's own "to fully utilize gpu memory"
  suggestion, see §5 — but that is budget-pinning, not edge-riding: on discrete
  VRAM the suggestion is trustworthy, unlike on GB10.)
- No `NCCL_P2P_DISABLE=1`: contrary to the glm-5.2-sm120 recipe's "mandatory",
  PCIe P2P is OK-listed (`nvidia-smi topo -p2p r`) and benched clean and fast.
- No instanttensor loader (v9): local NVMe load is already 26 s.
- No `--enforce-eager`: CUDA graphs (piecewise + full) capture and run clean —
  this was the single biggest perf lever (10x decode).

## 4. Exact launch

`./launch-glm53-vllm-sm120-tp4.sh` (defaults are the flagship config):

```bash
vllm serve /models/glm-5.3-flash-fp8 \
  --served-model-name glm-5.3-flash \
  --host 0.0.0.0 --port 8010 \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.95 \
  --kv-cache-memory=10934056960 \
  --max-model-len 262144 \
  --max-num-seqs 6 --block-size 2304 \
  --kv-cache-dtype fp8_e4m3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":4}' \
  --tool-call-parser glm47 --enable-auto-tool-choice \
  --reasoning-parser glm45 \
  --distributed-executor-backend mp
```

Docker env: `NCCL_P2P_DISABLE=0`, `TORCH_CUDA_ARCH_LIST=12.0`,
`FLASHINFER_CUDA_ARCH_LIST=12.0a`, GPUs `0,1,2,3`, `:8010` (`:8000` is taken
by another service on this box). Boot to serving: **~7 min** (26 s weights,
~3.5 min warmup/JIT/graph capture).

Engine receipts: `FLASHINFER_MLA_SPARSE_SM90` backend selected on cap-12,
`DEEPGEMM` Fp8 MoE backend, `nccl==2.30.7`.

## 5. Performance ladder (3 streaming runs, 200 tokens, temp 0, thinking off)

| Config | TTFT (median) | Decode (median) | KV pool (fp8) |
|---|---:|---:|---:|
| eager, P2P off, no MTP (day-1 safe boot) | 0.143 s | 10.53 tok/s | 987,011 tokens |
| CUDA graphs + PCIe P2P | 0.139 s | 107.67 tok/s | 953,250 tokens |
| **+ MTP-4, 262K (flagship)** | **0.059 s** | **171.74 tok/s** | **557,795 tokens (2.13x @262K)** |

16x decode over the safe boot; ~4.8x the Sparks' TP4 flagship (35.7 tok/s) on
the same day-0 model. MTP acceptance is *better* than on GB10: mean acceptance
length 3.22, per-position [0.83, 0.60, 0.45, 0.35], avg draft acceptance 55.6%
(GB10: 2.5–2.9, [0.74, 0.47, 0.27, 0.15]) — native FP8 weights likely help the
draft head agree with the target.

Pool update (2026-08-27 14:31 UTC, pool-only change, no perf re-bench): after
the flagship boot vLLM's CUDA-graph memory profiling reported 94.07/95.01 GiB
free at the 0.95 budget and suggested `--kv-cache-memory=10934056960` (10.19
GiB) "to fully utilize gpu memory" (or raise gmu to 0.9545). Pinned it in the
launcher: KV pool 957,909 → **1,428,980 tokens**, concurrency **5.45x** full
262K requests. On discrete VRAM this suggestion is safe to take verbatim — the
counterexample is GB10, where the same-looking number is not feasible
([GB10-KV-MEMORY-LADDER.md](GB10-KV-MEMORY-LADDER.md)).

## 6. Validation (flagship config)

- Greedy coherence: exact, clean ("17 × 23 = 391" with a difference-of-squares aside).
- Reasoning parser (`glm45`): monologue in `reasoning`, clean `content` (thinking ON default).
  With thinking off, GLM leaks reasoning-prose into content — same as sm121, ship thinking-on.
- Tool calling (`glm47`): structured `tool_calls`, `finish_reason: tool_calls`.
- Logprobs: all finite.
- Long-context: 33K-token needle retrieved exactly (17.8 s wall) — sparse
  indexer + kpool sound past the `index_topk=2048` boundary.

## 7. Open items

- FlashInfer `backend="cutlass"` / trtllm-gen MLA decode (Blackwell-native) —
  candidate next perf lever over the fa2 fallback.
- 1M context: model-native `max_position_embeddings=1048576`. With the pinned
  10.19 GiB pool (1,428,980 tokens) a single full 1M-token request now fits
  WITH MTP resident — `--max-model-len 1048576` is within pool limits, ~1.37x
  concurrency at full depth. Prefill wall-clock is the real cost. Untested.
- `num_speculative_tokens=3`: position-4 acceptance is 0.345 here (vs 0.15 on
  GB10), so MTP-4 earns its keep on sm120 — revisit only if batch throughput matters.
- Vision: `chat_template_mm.jinja` (repo) not yet mounted; the zai checkpoint
  ships the same text-only template as the NVFP4 quant.
- Upstreaming: all seven fixes remain unmerged upstream (PR #53906 still open).
