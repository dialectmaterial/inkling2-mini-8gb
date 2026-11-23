# inkling2-mini-8gb

Public write-up of running Inkling 2 Mini on an RTX 5060 Laptop (8 GB). Data and docs only, no code to build.

- Raw experiment directory (not in this repo): `~/.cache/llama.cpp/2026-11-23_inkling2-mini/` (configs, run logs, Neon reviews).
- Live service: `~/.config/systemd/user/inkling2-mini.service`; `systemd/` here is a scrubbed copy.
- Before committing: grep for the home username, key paths, emails and tokens; configs use `~` and never name the key file.
- Numbers in README come from `results/ledger.jsonl` (hill-climb harness) or `results/lab-gates*.json` (registry gates); say which.
