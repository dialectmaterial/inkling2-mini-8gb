# Inkling 2 Mini on an 8 GB laptop GPU: 23 -> 37 tok/s

[Thinking Machines' Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) (26B looped MoE) does not fit
in 8 GB of VRAM. This repo shows how it runs at **37 tok/s** on an **RTX 5060 Laptop (8 GB)** with 30 GB of RAM, measured
with the [inference-optimisation skill](https://github.com/sybil-solutions/inference-optimisation-skill) and checked with
the [local-ai-registry](https://github.com/sybil-solutions/local-ai-registry) lab gates.

## Results

| metric | before | tuned (stock llama.cpp) | tuned + MoE-cache fix |
|---|---:|---:|---:|
| decode, single user (tok/s, median of 6 x 512 tokens) | 23.1 | 26.8 | **37.2** |
| prefill, 3.3K-token prompt (tok/s) | 104 | 121 | **174** |
| prefill, registry context gate, 22K tokens (tok/s) | - | 53 | **106** |
| decode at 30.5K context (tok/s) | - | 26.7 | **32.1** |
| needle retrieval at 30.5K of 32K | - | 3/3 | **3/3** |
| registry lab gates (load, chat, reasoning, tools, context, speed) | - | 6/6, 26.9 tok/s | **6/6, 37.9 tok/s** |
| peak VRAM (GiB) | 7.20 | 7.33 | 7.35 |

- **before**: experts of some blocks pinned to the CPU by hand (`-ot`), the best static layout that fits.
  Stock llama.cpp defaults (`-c 32768`, auto-fit) do not start at all: they ask for a 10.4 GB compute buffer.
- **tuned**: all experts in RAM, a 5.6 GB GPU cache for the hot ones (`--cpu-moe --moe-cache-mib 5600`).
- **+ fix**: llama.cpp's MoE cache treated the 3 passes of Inkling's looped blocks as separate layers, so it stored and
  uploaded the same experts up to 3 times ([issue #32744](https://github.com/ggml-org/llama.cpp/issues/32744)).
  The fix from that issue (by adishik, branch `moe-cache-shared-experts`, commit `65dd939f3`) shares one cache entry per expert.
- Every candidate ran between two controls and was kept only if it beat both. Full log: `PLAN.md`; every run: `results/ledger.jsonl`.
- **Prefill depends on the prompt.** With experts in RAM, speed depends on how many different experts a prompt touches:
  the harness's random-word filler and the registry's numbered-line filler differ by about 2x.

## Run it

Weights: `unsloth/Inkling-2-Mini-GGUF` / `Inkling-2-Mini-UD-Q3_K_XL.gguf` @ `8078d149286eec7193795413b9e0643da6d11fa1` (13.6 GB).

Engine: llama.cpp `cbb1c1c` with the fix cherry-picked:

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout cbb1c1c50bba81929387dea473873d1df87a6212
git fetch https://github.com/adishik/llama.cpp moe-cache-shared-experts && git cherry-pick 65dd939f3
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120a
cmake --build build --target llama-server
```

```bash
llama-server --model Inkling-2-Mini-UD-Q3_K_XL.gguf --jinja \
  --n-gpu-layers 99 --flash-attn on --fit off --ctx-size 32768 --parallel 1 \
  --batch-size 512 --ubatch-size 32 --cache-type-k q4_0 --cache-type-v q4_0 \
  --no-op-offload --load-mode none --cpu-moe --moe-cache-mib 5600 \
  --threads 8 --threads-batch 12
```

Without the fix the same command gives the "tuned" column. `systemd/inkling2-mini.service` is the user unit used here.

Hardware: RTX 5060 Laptop 8 GB (measured 321 GB/s), Core Ultra 7 255H, 30 GB RAM (measured 74 GB/s), CUDA 13.4.

## What did not help

| tried | result | why |
|---|---|---|
| DSpark speculative decoding (llama.cpp PR #32218) | 17.9 tok/s (-22%) | verifying several tokens reads more experts from RAM; the draft also takes VRAM |
| bigger prefill batches (64, 128) | prefill about -50% | the MoE cache only serves batches of 32 tokens or fewer; larger ones fall back to the CPU |
| CPU computes 25% of cache misses | 19.4 tok/s | the CPU is slower than uploading the expert |
| 5.8 GB cache instead of 5.6 | +1% | inside noise, less VRAM headroom |
| other thread counts, batch 16 | no gain or worse | the GPU is the bottleneck once the cache is on |

## Output equivalence

With the fix, greedy (temperature 0) output is not bit-identical to stock llama.cpp. Stock llama.cpp also changes its greedy
output when the cache size changes (5600 vs 2800 MiB), and the fix's keying with stock slot counts
(`LLAMA_MOE_CACHE_DEDUP=2`) reproduces stock output exactly. Token probabilities stay close; the texts split at near-ties.

## Not verified

Vision; several users at once; contexts above 32K; answer quality against the full-precision model; the `q4_0` KV cache on real coding work.

## Context

The local-ai-registry accepts no GGUF recipes on NVIDIA, so this setup cannot be a registry recipe. For cards smaller than the
model, llama.cpp with a GPU expert cache is what works today.

Measured on 2026-11-23. Written with help from Claude Code (Anthropic) and reviewed by Gemini (Antigravity).
