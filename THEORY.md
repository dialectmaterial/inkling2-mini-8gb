> Note (2026-11-23 review): the decode byte counts below come from the earlier IQ3_S file; for UD-Q3_K_XL the active weights are ~3.65 GB per token, and the prefill line should read 48 x 90M + 0.46B = ~4.8B. The conclusions are unchanged. See PLAN.md.

# Ceilings: Inkling-2-Mini on RTX 5060 Laptop 8 GB (Yoga 9i Pro, Core Ultra 7 255H)

Measured 2026-11-23 (torch, scratch script): GPU device copy 321 GB/s (spec 384), GPU fp16 GEMM 34.6 TFLOPS,
CPU read 74 GB/s (12 threads). VRAM 7.62 GiB usable. Host RAM 30 GiB.

## Model (from GGUF, inkling2_loop)
24 blocks, hidden 2304, 128 experts top-8 (expert ffn 1152), GQA 24/8 x 128, SWA 1024 on most layers.
Loop: blocks 6-17 run 3 times (loop.count=3), so one token = 12 edge passes + 36 loop passes = 48 layer passes.
Bytes (uncensored IQ3_S file, official Q3_K_XL similar size):
- routed experts: 0.483 GB per layer (all 128), active 8/128 = 30.2 MB per layer pass
- non-expert per layer (attn, shared expert, norms): ~30-33 MB
- output (LM head) 0.49 GB, token_embd 0.49 GB (embedding lookup reads one row only)

## Decode (bytes per token / bandwidth)
Layout of the current unit: edge experts (blocks 0-5, 18-23) on CPU, loop experts on GPU.
- CPU: 12 x 30.2 MB = 0.36 GB / 74 GB/s = 4.9 ms
- GPU: 36 x 30.2 MB (loop experts) + 12 x 30 MB + 36 x 32.6 MB (non-expert) + 0.49 GB (head) = 3.1 GB / 321 GB/s = 9.7 ms
- KV at 32K: SWA layers cap at 1024 tokens; small next to weights.
- Serial ceiling (llama.cpp runs CPU and GPU parts in sequence): 14.6 ms = 68 tok/s. With perfect overlap: 103 tok/s.
- Putting loop experts on GPU is right: they are read 3x per token, edge experts 1x.
- Speculative decoding (DSpark draft, 0.5 GB Q8_0): with accepted length a, target bytes / a plus draft bytes.
  Verify of k tokens reads the union of experts chosen by k tokens, so MoE gains less than dense.

## Prefill (compute)
Active params per token: 48 passes x ~90M + 0.46B head = ~3.7B -> ~7.5 GFLOP/token.
At 50% of 34.6 TFLOPS: ~2300 tok/s if all weights were on GPU.
With CPU-resident experts: either CPU computes them (bound by CPU, ~100s of tok/s) or llama.cpp's op offload
streams the CPU experts to the GPU once per ubatch (5.8 GB over PCIe per ubatch; at ~25 GB/s ~0.23 s per ubatch,
so ubatch 2048 -> ~8900 tok/s bound, ubatch 512 -> ~2200 tok/s bound). The current unit has --no-op-offload
and ubatch 32, so prefill is CPU-bound at ~120 tok/s: the largest gap.
