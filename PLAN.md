# 2026-11-23 Inkling-2-Mini hill-climb, RTX 5060 Laptop 8 GB, llama.cpp cbb1c1c

Goal: replace the uncensored Inkling-2-Mini (AtomicChat AD-IQ3_S) with official weights
(unsloth/Inkling-2-Mini-GGUF UD-Q3_K_XL @8078d149), measure before (default launch) and after (tuned).
Constraints: VRAM cap 7.9 GiB (probe nvidia-smi), ctx 32768 must stay stable, single user, lossless only
(KV quant counted as a labelled trade, as the old unit already used q4_0).
Harness: ~/.claude/skills/inference-optimisation/scripts/bench.py (api_key_file local addition), temp 0.6,
reasoning on (template default), 6 x 512-token decode samples, median. AC, performance profile.
Server via launch.sh (transient unit inkbench, port 8086). qwen35-9b.service stopped during runs; restart after.

## Log
- 19:10 uncensored-current (old unit flags, AD-IQ3_S): c1 26.14 (26.03-26.45), prefill 124 @3.3K / 119 @12.9K,
  TTFT short 277 ms, startup 11.9 s, needle 11K 3/3, peak VRAM 7.39 GiB. Reference only.
- 19:31 official-default (`-c 32768`, llama.cpp defaults incl. --fit on): FAILS to start, compute pp buffer 10.4 GB
  (n_ubatch 512 x 32K ctx). Default launch is not usable on 8 GB.
- 19:33 official with old unit flags: FAILS, KV alloc OOM. CUDA0 model buffer 7.50 GB: unsloth keeps loop experts
  in NVFP4 (573 MB/block vs 483 MB in the IQ3_S file). Fix: block 12 (smallest loop block, 451 MB) experts to CPU.
  -> config "before" = old unit flags + block 12 on CPU. This is the before baseline for the official model.
- 19:40 block 12 only on CPU: OOM at cublas init. Blocks 12+17 on CPU fits, 7.35 GB VRAM -> config "before".
- 19:42 DSpark: unsloth dspark-Q8_0.gguf (new format) fails: joint_proj shape (build cbb1c1c expects old);
  dspark-Q8_0_Old.gguf fails: markov_w1 vocab 201024 vs 200058. Needs a newer llama.cpp. Deferred.
- 19:45 user reports other programs were running during earlier runs; GPU idle now. Reference re-run scheduled.
- 19:46 before: c1 23.12 (23.0-23.19), prefill 104 @3.3K / 101 @12.9K, needle 11K 3/3, startup 9.4 s. BASELINE.
- 19:53 threads6: c1 22.53 vs ctl 23.12 / 21.74 -> REJECT (not above both controls).
- 19:59 tb16 (threads-batch 16): prefill 110.6 vs ctl 101.5 / 103.8 (+6-9%), c1 22.96 flat -> ACCEPT.
- 20:05 tb6: prefill 77.9 -> REJECT.
  Note: fitctl-1 samples fell 23.1 -> 21.0 within the run (host activity); controls on both sides are needed.
- 20:08 llama.cpp has --moe-cache-mib (GPU cache for CPU experts): candidates mc5600, mc5600-cs25 prepared.
  Builds: A = PR #32218 (Inkling2 DSpark, e4b1cb5), B = A + #32883 (attention query chunking) + #32875 (NVFP4 dequant).
- 20:19 batch4 (prefill 4K only): bctl-1 23.17 / 111.7; best-a (build A, PR #32218) 23.01 / 111.2 -> engine neutral.
- 20:24 dspark-a (build A, DSpark Q8_0 draft n-max 3, block 13 also on CPU to fit; 7.6 GB VRAM): c1 17.94 (15.6-22.5),
  acceptance ~45%, mean len 2.35 -> REJECT. Verify of k tokens reads the union of experts on the CPU-resident
  blocks, and the draft costs a loop block. Speculation does not pay with 1/3 of the expert bytes in host RAM.
- 20:27 best-b (build B: + query chunking + NVFP4 dequant) at ub 32: 22.78 / 111.3 -> neutral at this ubatch.
- 20:30 mc5600 (--cpu-moe --moe-cache-mib 5600, no -ot): c1 26.96 (25.6-28.3), prefill 121.9, VRAM 7.33 GiB
  vs bctl-1 23.17 / bctl-2 22.89 -> +17% decode, +10% prefill. ACCEPT (repeat scheduled).
- 20:37 batch5: mcctl-1 26.98 / 121.4 (repeat confirms mc5600); mc5800 27.28 (+1%, inside noise, +0.19 GiB VRAM) -> REJECT;
  mc5000 25.56 / 113.7 -> REJECT (cache size matters ~+2.3 tok/s per GB); mc-cs25 (cpu share 0.25) 19.43 -> REJECT
  (CPU computing misses is slower than uploading them); mcctl-2 26.78 / 120.0.
- 20:52 batch6: mcctl-3 26.81/119.7; mc-tb16 26.97/119.5 -> REJECT (tb16 helped only when the CPU computed experts);
  mcb-tb16 (build B) 26.87/118.4 -> neutral, REJECT (stay on the old build); mcb-ub64 (cache 5300) 25.91 / 55.0 and
  mcb-ub128 (cache 4900) 25.02 / 58.8 -> REJECT: bigger ubatch touches more experts per step, so more cache misses
  are uploaded; prefill halves. mcctl-4 26.74/122.2.
  Three levers in a row inside noise or worse -> plateau.
- 21:08 mc-ub16: 26.74 / prefill 92.2 -> REJECT.
- 21:11 final (= mc5600): c1 26.82 (25.7-27.8), prefill 121 @3.3K / 129 @12.9K, long 11K: prefill 129, decode 25.4, 3/3.
- 21:17 needle at 23.5K (72%): 3/3, decode 24.4. 21:21 needle at 30.5K (93% of 32K): 3/3, prefill 117, decode 26.7,
  VRAM peak 7.33 GiB, no flags. QUALIFIED. Lossless vs before (same weights; cache only moves where experts live).

## STATUS (plateau)
best = mc5600: --cpu-moe --moe-cache-mib 5600, ub 32, threads 8/12, KV q4_0, build cbb1c1c.
before (official, old flags + blocks 12,17 on CPU): c1 23.12, prefill 104/101, needle 11K decode 22.56, TTFT 11K 108.8 s.
after: c1 26.82 (+16%), prefill 121/129 (+17%/+28%), needle 11K decode 25.4 (+13%), TTFT 11K 85.1 s (-22%).
Ceiling (THEORY.md): serial ~68 tok/s if the hot set fit; the cache size (VRAM) is the binding limit (~+2.3 tok/s per GB).
Remaining levers: newer llama.cpp once #32883/#32875 merge (neutral at ub 32 here); KV q8_0 (quality, costs cache);
smaller quant (UD-Q2_K_XL 9.1 GB) for a higher hit rate at some quality cost; DSpark only if weights fit in VRAM.

## 2026-11-23 21:50 Neon (Gemini 4 Neon High, read-only) review: NEON-REVIEW.md
Verified by Claude:
- MoE cache counts the 3 loop passes as separate layers: log says "5589.20 MiB for 24522.75 MiB of host experts"
  (real experts ~11.6 GB); inkling2-loop.cpp post_arch_tensors aliases layers, llama-moe-cache.cpp:423 iterates
  model.layers -> loop experts can be cached/uploaded up to 3x. Real bug; dedup patch = top lever.
- llama-moe-cache.cpp:369 max_batch = 32: ub > 32 bypasses the cache -> CPU prefill. The ub64/ub128 rejection
  reason above is wrong; op-offload + ub 256/512 on build B was never tested.
- THEORY.md numbers used the IQ3_S file and has an arithmetic slip (48 x 90M + 0.46B = 4.8B, not 3.7B).
Not accurate in the review: tool round trip WAS tested on the installed unit (21:30 smoke test, quoted ZX-4471);
power profile was already "performance" before this session (not changed).
Fixed: inkling2-mini stopped, qwen35-9b restarted (pi refuses its default model while inkling runs unmanaged).
- 22:0x registry lab gates (local-ai-registry lab/lab.py gates(), called directly; no llama.cpp engine profile exists,
  so no recipe written) on inkling2-mini.service: ALL 6 PASS (load chat reasoning tools context speed),
  speed 26.9 tok/s (30 s window, server tokenizer). Context gate: 22,250-token prompt in 423.5 s = 53 tok/s, and the
  server log agrees (418.9 s prompt eval). My harness got 117-129 tok/s on its random-word filler: prefill depends
  on the prompt (expert routing -> cache misses). The 121-129 prefill numbers hold for that filler only. Investigate.

## 2026-11-23 22:30 MoE-cache loop dedup (upstream issue #32744, fix from adishik:moe-cache-shared-experts 65dd939f3)
Cherry-picked onto cbb1c1c as branch moe-cache-loop-dedup (~/.local/src/llama-moefix), build ~/.local/share/llama.cpp/ink-dedup.
Startup: "5587.42 MiB for 11633.62 MiB of host experts" (was 24522.75). Decode hit rate 93.97%.
Greedy (temp 0, 3 prompts x 300 tokens): unfixed is deterministic (off == off). Fix mode 2 (shared keys, old slot counts)
== unfixed exactly -> keying exact. Mode 1 (default) differs from char 15-135; logprobs close (top token 99.1% vs 98.9%),
split at near-ties. Unfixed at --moe-cache-mib 2800 also differs from 5600 -> output depends on bank shape in stock
llama.cpp too; not caused by the fix. Verdict: correct.
- 22:44 batch8: ddctl-1 26.77/120.1; dd-on (fix) 37.15 (36.3-38.5) / 174.1; dd-off (same build, LLAMA_MOE_CACHE_DEDUP=0)
  26.80/119.1; ddctl-2 26.41/118.3 -> ACCEPT: +39% decode, +45% prefill, VRAM 7.35 GiB unchanged.
- 22:53 dd-needle: 30.5K prompt 3/3, prefill 149, decode at depth 32.1 (was 117 / 26.7). 7.33 GiB, no flags.
- 23:00 registry lab gates on the fix: ALL 6 PASS, speed 37.9 tok/s, context-gate prefill 106 tok/s (was 53).
- 23:05 INSTALLED: inkling2-mini.service -> ~/.local/share/llama.cpp/ink-dedup (LD_LIBRARY_PATH set: binary RUNPATH
  points at the source build dir). Previous unit: inkling2-mini.service.pre-dedup. Smoke test OK.
Neon design (NEON-FIX-DESIGN.md) matches the patch on the scheduler trap (per-pass slot map tensors) and the clock
(patch ticks only on the key pass). Its follow-up idea: weight cache budget by passes (n_refs), est. +1-3 tok/s.
