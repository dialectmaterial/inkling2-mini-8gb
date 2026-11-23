# inkling2-mini-8gb

Public write-up of running Inkling 2 Mini on an RTX 5060 Laptop (8 GB). Data and docs only, no code to build.

- Raw experiment directory (not in this repo): `~/.cache/llama.cpp/2026-11-23_inkling2-mini/` (configs, run logs, Neon reviews).
- Live service: `~/.config/systemd/user/inkling2-mini.service`; `systemd/` here is a scrubbed copy.
- Before committing: grep for the home username, key paths, emails and tokens; configs use `~` and never name the key file.
- Numbers in README come from `results/ledger.jsonl` (hill-climb harness) or `results/lab-gates*.json` (registry gates); say which.

## Lessons (2026-11-23)
- Benchmark with controls on both sides; host activity can drift decode by ~8% inside one run.
- Stop systemd units one by one: one `systemctl stop a b c` gives up if any unit is not loaded.
- MoE cache only serves ubatches of 32 tokens or fewer; larger ubatches fall back to CPU prefill.
- The MoE-cache loop fix is a community patch (llama.cpp #32744); llama.cpp forbids AI-written PRs and comments, so do not submit it.
- Prefill depends on prompt content; quote which filler a prefill number came from.
