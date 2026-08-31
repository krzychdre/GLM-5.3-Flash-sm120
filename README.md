# GLM-5.3-Flash on SM120 — 4x NVIDIA RTX PRO 6000 Blackwell

**This repo is a fork of the [DGX Spark deployment](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-2x-DGX-Spark) (GB10, SM121), re-targeted at SM120: a single x86_64 node with 4x RTX PRO 6000 Blackwell Workstation 96 GB.** The Spark work proved that GLM-5.3-Flash needs seven day-0 fixes across vLLM/FlashInfer on any capability-12 silicon; this fork re-verifies each bug on SM120 with the same probe scripts, ports the patch stack to amd64 (`docker/Dockerfile.glm53-sm120`), and serves the **native FP8 checkpoint** (328 GB — no quantization) instead of NVFP4:

| | this fork (SM120 TP4) | upstream (2x DGX Spark TP2) |
|---|---|---|
| Decode / TTFT | **171.7 tok/s / 0.059 s** | 21.8 tok/s / 0.289 s |
| Weights | zai-org native FP8 | NVFP4 weight-only quant |
| Context / KV pool | 262K / 1,428,980 fp8 tokens (5.45x) | 262K / 672,606 fp8 tokens |
| Launcher | `launch-glm53-vllm-sm120-tp4.sh` | `launch-glm53-vllm-tp2.sh` |

Full port report with the bug-by-bug probe results: [docs/SM120-DEPLOY-REPORT.md](docs/SM120-DEPLOY-REPORT.md). Everything below the RTX 6000 section documents the original DGX Spark deployment, kept intact as reference — the root-cause analyses there are what made the sm120 port a same-day job.

## RTX PRO 6000 Blackwell — the hardware this fork targets

- **GB202** (consumer/workstation Blackwell), compute capability **12.0 (sm_120)** — same `major==12` family as the Spark's GB10 (sm_121), which is why the whole sm121 patch stack applies with only `12.1a` → `12.0a` arch-list changes.
- **96 GB GDDR7 ECC**, 512-bit bus, ~1.8 TB/s bandwidth per card — 4 cards = **384 GB VRAM**, enough to shard the full FP8 checkpoint (76.4 GiB/rank) with ~11 GiB/rank to spare.
- 24,064 CUDA cores, 5th-gen Tensor Cores with native **FP8 and FP4** — the fp8 KV cache and DeepGEMM fp8 MoE path run on native hardware, no emulation.
- **No NVLink** — TP4 runs over PCIe Gen5 **P2P**, which is OK-listed and fast on this platform (`nvidia-smi topo -p2p r`); the glm-5.2-sm120 recipe's "P2P must be disabled" did **not** reproduce here.
- Discrete VRAM (vs GB10's unified memory): vLLM's memory profiling is truthful — no phantom KV allocations, no cache-flush ritual. `--kv-cache-memory` is only used as budget-pinning with vLLM's own "fully utilize" suggestion (10.19 GiB → 1,428,980 fp8 tokens), which is safe here but not feasible on GB10.
- Measured under decode load: ~90% GPU util, ~300–345 W/card (600 W TBP headroom), ~5 CPU cores host-side (scheduler + NCCL/CUDA spin-wait).

---

# See the Best repo for RTX 6000 Pro runbooks (for many different models/quants):

https://github.com/local-inference-lab/rtx6kpro/blob/master/models/glm-5.3-flash.md
---

# Original project: GLM-5.3-Flash on 2x NVIDIA DGX Spark

Serving [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (320B total / 18B active MoE, released 2026-08-26) across two DGX Spark (GB10, SM121) nodes at tensor-parallel 2, using the [LibertAIDAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4) weight-only NVFP4 quant. **262,144-token context on TP2 — and up to the model-native 1,048,576 (1M) on TP4, whose 1.26M-token KV pool holds a full 1M-token request. Working, benchmarked, same-day as the model drop.**

As far as we can tell this was the first working GLM-5.3-Flash deployment on DGX Spark hardware. Getting there took fixing **seven distinct day-0 bugs** across vLLM, FlashInfer, and their dependency chain — every one is documented in [docs/DEPLOY-REPORT.md](docs/DEPLOY-REPORT.md) with root causes, receipts, and the probe scripts that found them.

## Results

| Metric | bf16 TP2 (v7) | fp8+MTP-4 TP2 (v8) | **fp8+MTP-4 TP4 (v8, flagship)** |
|---|---|---|---|
| TTFT (median, 3 runs) | 0.239 s | 0.289 s | **0.204 s** |
| Decode | 14.3 tok/s | 21.8 tok/s | **35.7 tok/s (peak 36.8)** |
| Context | 262,144 | 262,144 | **up to 1,048,576 (model-native 1M)** |
| KV pool | 603,144 tokens | 672,606 tokens (local weights) | **1,263,415 tokens (4.82x full-context)** |
| Nodes | 2 | 2 | 4 (`launch-glm53-vllm-tp4.sh`) |
| Boot time | ~14 min | ~21 min | **~12 min** (quarter weights/rank load faster) |

TP4 also dissolves the GB10 KV-allocation ceiling documented in the memory-ladder study: at ~50 GiB weights per rank the 9 GiB KV slab allocates with ~60 GiB of slack — the 1M+ token pool that TP2 physically could not hold.

**1M context on TP4:** GLM-5.3-Flash ships `max_position_embeddings = 1,048,576`, and the TP4 KV pool (1,263,415 tokens) exceeds one full-length request, so `--max-model-len 1048576` is within both the model's and the pool's limits — no rope scaling, no overrides. The launcher now defaults to 1M. Practical notes: a full 1M-token prefill takes many minutes of wall clock before the first output token, and concurrency at full depth is ~1.2 requests; cap `--max-model-len` lower (e.g. 300000) when you want a snappier multi-user endpoint.

**TP2 KV update (2026-08-27):** with LOCAL weights on both ranks (no NFS duty) plus an aggressive cache-flush ritual, TP2 holds **672,606 fp8 KV tokens** (`--kv-cache-memory 5905580032`), stress-verified — a +33% jump over the 507,041-token NFS-bound ceiling (which still applies when one rank doubles as the NFS server). The Results table above now reflects this local-weights number. 6 GiB+/rank reservations "succeed" then die on first touch in warmup (phantom backing). The 8-attempt hunt, the first-touch failure signature, and every lever that did NOT work are in [docs/KV-HUNT-672K-TP2-RECORD.md](docs/KV-HUNT-672K-TP2-RECORD.md).

MTP metrics under real traffic: mean acceptance length 2.5–2.9, per-position acceptance ~[0.74, 0.47, 0.27, 0.15] — position 4 is nearly free-riding, so `num_speculative_tokens=3` is a candidate micro-tune.

## Why this needs a patched image

The vLLM PR authors' day-0 image (`vllm/vllm-openai:glm53-flash-arm64-cu130`) works on B200. On GB10/SM121 it fails five separate ways. Our derivative (`docker/Dockerfile.glm53-sm121*`, applied in order v1→v7) fixes:

1. **NoPE MLA vs the SM12x sparse backend** — the only stock capability-12 sparse-attention backend requires the packed `fp8_ds_mla` cache layout, which hardcodes DeepSeek's `pe_dim=64`. GLM-5.3 is NoPE (`qk_rope_head_dim=0`) → assert death in warmup. Fix: extend vLLM's SM90 NoPE sparse-MLA backend (FlashInfer `BatchMLAPagedAttentionWrapper`, plain bf16 cache) to SM121 with the FA2 path — probed directly on the GPU with the model's real shape before trusting it (`probes/probe_sm121_nope_mla.py`).
2. **FlashInfer 0.6.17 FA2 MLA NaN** — the FA2 scheduler produces NaN for 64–256-row batches on SM121 (bisect: `probes/probe_fa2_bisect.py`). Normal prompts land exactly there. Fix: FlashInfer **0.6.18 nightly**.
3. **The nightly's dependency sabotage** — installing it silently downgrades `nvidia-nccl-cu13` to 2.29.7 (NCCL "internal error" on the Spark IB fabric; re-pin **2.30.7**) and skews `nvidia-cutlass-dsl` to a mixed 4.7.0/4.6.2 install (CuTeDSL warmup ICE; re-pin **4.6.2**). Audit transitive pins after ANY pip install in these images.
4. **PDL on unvalidated silicon** — vLLM enables Programmatic Dependent Launch for capability ≥ 9, including SM121, in the Triton kernels carrying KDA recurrent state. Gated off on SM12x.
5. **Indexer uninitialized top-k** — the kpool top-k destination was `torch.empty` and the kernels only guarantee the first `min(k, valid)` entries; short rows carried garbage pool ids → bogus token indices → attention gathers uninitialized KV → NaN lottery. Fix: init to `-1` + clamp expanded pool ids (`docker/patch_v7.py`).
6. **fp8 KV cache unlock** (v8, `docker/patch_v8_fp8.py`) — see the fp8 section below. Also learned the hard way: MTP's draft head (+~5GB) at gmu 0.85 trips GB10 unified-memory OOM (`NV_ERR_NO_MEMORY` in dmesg, death on first request) — pin the budget with `--kv-cache-memory` (vLLM prints the safe number in its startup log) instead of riding the gmu edge.

Two serve-flag landmines (no code needed):
- `--block-size 2304` — vLLM's hybrid block aligner picks a size whose kpool storage tiles by 32, but DeepGEMM's arch-12 fp8 paged-MQA accepts only 64-entry pool pages. 2304 is a multiple of kpool·64 and of the MLA 128 alignment.
- `--gpu-memory-utilization 0.85` — 0.78–0.80 starve the bf16 KV cache at 131K+. (Credit: barrydeen's independent recipe.)

## Quickstart

Nodes: head owns the weights and serves `:8000`; worker mounts them over NFS at the same path.

```bash
# 1. Build the patched image on the head, ship to the worker
docker build -f docker/Dockerfile.glm53-sm121 -t glm53:sm121-v1 .   # then v2..v7 in order, each FROM the previous
docker save glm53:sm121-v7 | ssh worker docker load

# 2. Pre-launch ritual (BOTH nodes, every launch — GB10 unified memory)
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches
# worker: verify the NFS mount answers: ls /var/tmp/glm-5.3-flash-nvfp4/config.json

# 3. Launch worker FIRST, wait ~25 s, then head
./launch-glm53-vllm-tp2.sh 1   # on the worker
./launch-glm53-vllm-tp2.sh 0   # on the head
```

Edit the top of `launch-glm53-vllm-tp2.sh` for your IPs/paths. Full serve args, NCCL fabric env, and rationale for every flag: [docs/DEPLOY-REPORT.md](docs/DEPLOY-REPORT.md).

Smoke test:
```bash
curl http://<head>:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.3-flash","messages":[{"role":"user","content":"Say hello and name yourself."}],"max_tokens":40,"chat_template_kwargs":{"enable_thinking":false}}'
```

## Fast loading: InstantTensor (added 2026-08-27)

**Status: experimental — measured 15x load speedup, but NOT stable in our multi-node TP2 topology.** The v9 image adds the InstantTensor direct-I/O loader (`--load-format instanttensor`): loads drop from ~10 minutes to 40-100 seconds and the page cache stays empty. However, in ALL four of our v9 TP2 boots a rank died silently ~1 minute after loading (exit code None, nothing in dmesg) — at every KV budget, including the otherwise rock-stable 4.14 GiB — so the shipped launchers do NOT enable it and the stable image remains v8. This matches the known multi-node instability class for direct-IO loaders on Spark (eugr/spark-vllm-docker#29 reports fastsafetensors hanging in cluster mode). Single-node or TP4 use may fare better; re-test when upstream moves. Other things to know: its pip install silently downgrades NCCL to a fabric-fatal version (v9 re-pins 2.30.7 in the same layer), and because direct I/O never fills the page cache, it also defeats the first layer of the GB10 KV-allocation wall -- the full story and the remaining (unsolved) second wall are in [docs/GB10-KV-MEMORY-LADDER.md](docs/GB10-KV-MEMORY-LADDER.md). Credit: jack6464 (NVIDIA forum) for the pointer.

## Hard-won operational rules

- **Tear down BOTH ranks before relaunching either.** A new rank that rendezvouses with a dying one hangs or dies confusingly.
- **`grep '^IMAGE' launch-*.sh` on BOTH nodes before every launch.** Two of our "mystery" garbage boots were a silent image-version mismatch between ranks (in-place remote edits had failed silently). Copy whole files between nodes, never sed over ssh.
- **Capture `docker logs` before `docker rm -f`.**
- Two consecutive unexplained deaths = stop and diagnose, never crash-loop.
- `max_tokens` includes reasoning tokens when thinking is on; pass `chat_template_kwargs: {"enable_thinking": false}` per-request to disable.

## fp8 KV cache on GB10: a world first, and it's a two-line fix

FlashInfer gates fp8 MLA KV to SM90, and naively relaxing the gate fails with CUDA "invalid argument" (`probes/probe_fa2_fp8.py`). The actual root cause (`docker/patch_v8_fp8.py`): the fa2 fp8 branch **forces CTA_TILE_KV=32 — a Hopper 228KB-smem assumption**. On GB10's ~101KB opt-in max that doubles the tile picked for this device and over-requests shared memory (117,312B > 101,376B) at `cudaFuncSetAttribute`, before the kernel ever launches. **Capping the tile instead of forcing it** (fp8 keeps TKV=16 on 100KB-class devices — 91,680B, fits) plus the gate relax makes fp8 KV work: verified on-GPU with all batch shapes clean and rel-err ~0.005 vs an fp32 reference (normal fp8 quantization noise), then validated end-to-end serving GLM-5.3 with MTP.

As far as we can tell this is the first fp8 KV cache for a NoPE-MLA model on any consumer Blackwell part (GB10 or RTX PRO 6000 SM120 — see `docs/issue-flashinfer-fp8-mla-sm121.md` and `docs/issue-vllm-nope-fp8-ds-mla.md`, upstream-ready issue drafts with the receipts).

See also [docs/GB10-KV-MEMORY-LADDER.md](docs/GB10-KV-MEMORY-LADDER.md) — the six-boot study of why KV budgets above vLLM's suggested number die on GB10, with the cache-flusher mitigation ([cache_flusher.sh](cache_flusher.sh)) and the driver-level mechanism.

Phase-3 option (documented, not built): a ~40-line "zero-pad rope" shim would route NoPE models onto the Blackwell-native trtllm packed-fp8 decode kernel — faster decode, needs one FlashInfer kernel re-instantiation for GLM's 2176-wide kpool index buffer.

## Debugging kit (reusable for any day-0 model on new silicon)

- `probes/probe_sm121_nope_mla.py` — probe a FlashInfer kernel with your model's real geometry BEFORE patching arch gates.
- `probes/probe_fa2_bisect.py` — NaN bisect harness over batch shapes.
- `probes/probe_mhc.py` — A/B a Triton/TileLang kernel vs its torch reference.
- The deploy report describes the env-gated forward-hook NaN localizer (`GLM53_NAN_DEBUG=1` build) that names the first module emitting non-finite values — how we localized both NaN sources.
- `probes/bench_glm53.py` — 3-run TTFT/decode benchmark.

## vLLM v0.28.0 status (checked 2026-08-27)

**Not viable for GLM-5.3 yet**: the `glm5_next` architecture is NOT in the v0.28.0 release
(PR vllm-project/vllm#53906 still open/unmerged at check time), and no rebased day-0 image
exists (all `vllm/vllm-openai:glm53-flash*` tags still date to the original 2026-08-26 push).
The day-0 image used here is itself a main-branch dev snapshot (`0.1.dev20051`) cut around
the 0.28 branch point -- i.e. this stack already runs 0.28-era engine code plus the GLM
support 0.28 lacks. Upgrade path when it opens: watch the PR and the Docker Hub tags; the
patch stack here is guarded string-patches that apply-or-refuse loudly, so porting to a new
base is mechanical (apply v1->v10 in order, fix whichever guards fire, ladder through the
experiment lane before production).

## Credits

- Model: [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) · Quant: [LibertAIDAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4) (their sm_121 notes were used directly)
- **barrydeen** — the gmu 0.85 reference config and quantization-coverage table from their independently published DGX Spark recipe
- vLLM [PR #53906](https://github.com/vllm-project/vllm/pull/53906) authors for the day-0 image; FlashInfer for the 0.6.18 SM90-NoPE MLA path
- Deployed and debugged by Knox (Claude) for [@tonyd2wild](https://github.com/tonyd2wild)
