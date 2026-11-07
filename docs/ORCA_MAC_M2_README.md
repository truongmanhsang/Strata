# Orca on a Mac M2 (32 GB): build, run, use

This guide runs `orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF` (IQ3_XXS, two shards, 81.3 GB together) with Strata's
Metal engine on an Apple M2 Max with 32 GB of memory. The setup is manual. `strata launch` does not work with this
model: the launcher reads the file as Swift IQ3_S. For the general notes, see [ORCA.md](ORCA.md).

What you get:

- A chat server on `http://127.0.0.1:8080` with an OpenAI-compatible API and a web page.
- A 256K context window (262,144 tokens), with the KV cache in q4_0.
- Text only. Pictures are off. Tested with one request at a time.

Every number in this guide was measured on one machine. See [Speed and memory](#speed-and-memory-measured).

## 1. What you need

- An Apple Silicon Mac with 32 GB of unified memory. Tested: M2 Max, 32 GB, internal SSD, Darwin 27.2.0.
- About 170 GB of free disk at the peak: the two shards (81 GB), then the merged file (81 GB). You can delete the
  shards after the merge (section 4). The pack adds 1.4 GB.
- Xcode Command Line Tools, `cmake` and `ninja`.
- Python 3.10 or newer. Tested with 3.14.
- Homebrew `llama.cpp` for the `llama-gguf-split` tool. Tested with version 0.5.0.

## 2. Build the engine

This checkout has five source changes that are not committed yet. They add the dense GEMV and prefill kernels for the
IQ3_XXS and IQ2_S tensors that this model uses:

- `metal/kernels/gemv.metal`: adds IQ3_XXS and IQ2_S to the GEMV types.
- `metal/kernels/dense_mm.metal`: adds IQ3_XXS and IQ2_S to the dequantize-to-fp16 kernels.
- `metal/src/engine/window.cpp`: `pg_unit` gives IQ3_XXS and IQ2_S the 16-value unit.
- `metal/tests/gemv_parity.cpp` and `metal/tests/dense_mm_parity.cpp`: add the two types to the test lists.

Without them the server stops on the first request with `mtl: no function gemv_18_16_1_4_1 ...`.

Build the engine. The script runs the cmake and ninja steps and writes `build-metal/strata`. It uses `/usr/local/bin/cmake`
and `/usr/local/bin/ninja` and the path `/Volumes/DATA/Projects/strata`. Edit those at the top if your paths differ:

```bash
cd /Volumes/DATA/Projects/strata
sh build-strata-metal.sh
```

Set up Python:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Build and run the two parity tests. Both should end with `ALL PASS`:

```bash
ninja -C build-metal gemv_parity dense_mm_parity
build-metal/gemv_parity
build-metal/dense_mm_parity
```

## 3. Download the model

Download the two IQ3_XXS shards from Hugging Face, `orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF`:

- `Qwen3.8-Flash-Next-Uncensored-IQ3_XXS-00001-of-00002.gguf` (42.6 GB)
- `Qwen3.8-Flash-Next-Uncensored-IQ3_XXS-00002-of-00002.gguf` (38.7 GB)

Put both in one folder, for example `/Volumes/DATA/Models/Qwen3.8-Flash-Next-Uncensored-IQ3_XXS/`. Wait until both are
complete before you go on.

## 4. Merge the two shards into one file

The Metal engine reads one GGUF file. Shard 2 holds the experts for layers 13 to 47, so the engine needs both shards in one
file:

```bash
MODEL="/Volumes/DATA/Models/Qwen3.8-Flash-Next-Uncensored-IQ3_XXS"
/opt/homebrew/Cellar/llama.cpp/0.5.0/bin/llama-gguf-split --merge \
  "$MODEL/Qwen3.8-Flash-Next-Uncensored-IQ3_XXS-00001-of-00002.gguf" \
  "$MODEL/Qwen3.8-Flash-Next-Uncensored-IQ3_XXS.gguf"
```

Notes:

- The result is one file of 81.3 GB with 1224 tensors. The tool prints `merged from 2 split with 1224 tensors`.
- `--dry-run` does not write the file. Run the command without it.
- Do not use `--delete-splits`. The tool warns that it can leave you in an unrecoverable state.
- Before you delete the shards, check the merged file. We compared the tensor names, shapes, types and sizes with the
  shards, and compared sample tensors byte for byte. All matched.
- Keep the file on the internal SSD. We did not test an external disk.

## 5. Build the pack

The pack holds the dense weights and the tokenizer. Build it from the merged file. A pack built from shard 1 does not
match the merged file: the offsets differ, and the server stops with `native_experts.txt does not match the model`.

```bash
cd /Volumes/DATA/Projects/strata
STRATA_GGUF_PY="$PWD/third_party/llama.cpp/gguf-py" .venv/bin/python tools/iq_pack.py \
  --gguf "/Volumes/DATA/Models/Qwen3.8-Flash-Next-Uncensored-IQ3_XXS/Qwen3.8-Flash-Next-Uncensored-IQ3_XXS.gguf" \
  --out packs/orca-iq3_xxs-merged --compat-bf16
```

The output is 1.4 GB. `--compat-bf16` rounds the small projections to BF16, as [ORCA.md](ORCA.md) explains.

## 6. Optional: MTP draft head

The fastest setup in section 7 does not use MTP. We measured MTP as slower on this machine (12.2 tok/s against 13.9 tok/s,
both with `STRATA_METAL_TAIL_SKIP=7`). Skip this section unless you want to test it:

```bash
.venv/bin/python tools/mtp_fetch.py fetch --out mtp
.venv/bin/python tools/mtp_pack.py --src mtp --experts q2_0 --out mtp/mtp-q2_0.gguf
.venv/bin/python tools/mtp_rt.py --gguf mtp/mtp-q2_0.gguf --out mtp/rt
cp data/draft_vocab.bin mtp/rt/draft_vocab.bin
```

The fetch step writes about 4.9 GB of raw tensors to `mtp/tensors/`. You can delete that folder after packing.

## 7. Server config

The config is `strata-orca-fastG.json` in the repo root. Create it with this content. If your paths differ, change the
two model paths and the tokenizer path:

```json
{
  "exe": "build-metal/strata",
  "args": [
    "--pack", "packs/orca-iq3_xxs-merged",
    "--native", "/Volumes/DATA/Models/Qwen3.8-Flash-Next-Uncensored-IQ3_XXS/Qwen3.8-Flash-Next-Uncensored-IQ3_XXS.gguf",
    "--ple-gguf", "/Volumes/DATA/Models/Qwen3.8-Flash-Next-Uncensored-IQ3_XXS/Qwen3.8-Flash-Next-Uncensored-IQ3_XXS.gguf",
    "--expert-profile", "data/expert-profile.bin",
    "--expert-cache", "auto",
    "--spec", "4",
    "--spec-min-p", "0.5",
    "--max-context", "262144",
    "--kv", "q4_0",
    "--conversation-cache-mib", "0",
    "--prefill", "auto"
  ],
  "cwd": ".",
  "tokenizer": "packs/orca-iq3_xxs-merged/tokenizer",
  "model_name": "orcarouter-qwen3.8-flash-next-uncensored-iq3_xxs",
  "log": "strata-orca-fastG.log",
  "backend": "metal",
  "host": "127.0.0.1",
  "port": 8080,
  "env": {
    "STRATA_METAL_ARENA_GB": "10",
    "STRATA_METAL_PROMPT_CHUNK": "4096",
    "STRATA_METAL_TAIL_SKIP": "5",
    "STRATA_METAL_MOE_TILES": "8,1,4,4,2,4",
    "STRATA_METAL_GEMV_TILES": "m2",
    "STRATA_METAL_F16": "1",
    "STRATA_METAL_F16_STREAM": "1",
    "STRATA_METAL_Q3KR": "1",
    "STRATA_PIN_BY_NEED": "1"
  }
}
```

What the main settings do:

- `--max-context 262144 --kv q4_0`: the 256K window, with the KV cache in q4_0.
- `--spec 4 --spec-min-p 0.5` with no `--mtp`: drafting by lookup only. No MTP and no DFlash2.
- `STRATA_METAL_TAIL_SKIP=5`: skips rare expert reads. Decode is about 20% faster than with 7. Answers change more than
  with 7.
- `STRATA_METAL_PROMPT_CHUNK=4096`: prompt chunk size. 8192 makes prefill about 20% faster, but raises swap by about
  3.5 GB on this machine.
- `STRATA_METAL_ARENA_GB=10`: the GPU expert arena. 12 gave 10% more decode and 1.2 GB more swap.
- `--conversation-cache-mib 0`: turns off the conversation cache, so the RAM goes to the model.
- `--prefill auto` and `--spec` are both required. The server stops without them.

## 8. Start and stop the server

Start the server from the repo root:

```bash
cd /Volumes/DATA/Projects/strata
.venv/bin/python -m serve.server --engine strata --config strata-orca-fastG.json --port 8080
```

Keep the window open. The first start loads the experts into memory and takes several minutes. Check that the server is
ready:

```bash
curl -s http://127.0.0.1:8080/v1/models
```

It should list `orcarouter-qwen3.8-flash-next-uncensored-iq3_xxs` with `"status": {"value": "loaded"}`.

Stop the server and its engine:

```bash
pkill -f "serve.server --engine strata"
```

Engine logs go to `strata-orca-fastG.log` in the repo root. Server output goes to the terminal where you started it.

## 9. Use it

- **Web page:** open `http://127.0.0.1:8080`.
- **OpenAI-compatible API:** base URL `http://127.0.0.1:8080/v1`. The model id is
  `orcarouter-qwen3.8-flash-next-uncensored-iq3_xxs`.
- **Anthropic-compatible API:** `http://127.0.0.1:8080/v1/messages`.

Example request:

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"orcarouter-qwen3.8-flash-next-uncensored-iq3_xxs","messages":[{"role":"user","content":"Hello"}],"max_tokens":800,"temperature":0}'
```

The model thinks before it answers, and the thinking uses up `max_tokens`. With `max_tokens` 300 on an essay request, the
answer came back empty. Use 800 or more.

**oh-my-pi (`omp`):** the config is in `~/.omp/agent/`. Point `models.yml` at `http://127.0.0.1:8080/v1` with the model id
above and `contextWindow: 262144`. In `config.yml`, set `modelRoles.default` to
`strata/orcarouter-qwen3.8-flash-next-uncensored-iq3_xxs`. Start the server first.

## Speed and memory (measured)

Machine: M2 Max, 32 GB, internal SSD. Greedy decoding (`temperature` 0). The numbers come from one machine with one or two
runs each, so expect about ±10%.

**Decode** (tokens per second; a Vietnamese essay, `max_tokens` 800; some runs stopped earlier, which does not change tok/s
much):

| Setting | Decode tok/s |
| --- | --- |
| This guide (config G: TAIL_SKIP 5) | 18.7, 18.3 |
| TAIL_SKIP 5, arena 10 GB, MTP off | 18.5, 18.4 |
| TAIL_SKIP 7 (config C) | 15.5, 15.3 |
| TAIL_SKIP 7, arena 12 GB | 17.0, 16.9 (swap +1.2 GB) |
| TAIL_SKIP 7, `--spec 6` | 14.6, 15.2 |
| TAIL_SKIP 7, 12 I/O threads | 14.9, 14.8 |
| TAIL_SKIP 7, MTP on | 12.2 (one run) |
| TAIL_SKIP 7, DFlash2 on | 11.0 (one run; 63 of 224 drafts accepted, 28%) |

**Prefill** (prompt tokens per second; about 7,000 tokens, a new prompt for each run so that no prefix is reused):

| Setting | Prefill tok/s |
| --- | --- |
| Chunk 4096 (this guide) | 195, 217, 235 (about 215) |
| Chunk 8192 | 236, 262, 251, 273 (about 255); swap +3.5 GB on the first long prompt |
| 12 I/O threads, or CPU share off | about 220, about 219 (no real change) |

Short prompts (30 to 70 tokens) only reach 5 to 7 tok/s, because fixed costs dominate.

**Memory:** the model does not fit in RAM, so the experts stream from the SSD. Swap in use was between about 4.6 and
9.6 GB during our runs. Close other large apps before you start the server.

## Limits and what is not tested

- **Quality:** we checked a few prompts only: a short sum, a science question (why the sky is blue), and Vietnamese
  essays over several runs. There is no quality benchmark. `STRATA_METAL_TAIL_SKIP=5` changes answers more than 7. This is the
  engine's own rule.
- **256K:** the server reports the 262,144-token window, but we sent prompts of up to about 7,000 tokens. A full 256K
  prompt is not tested.
- **One request at a time:** we did not test several requests at once.
- **Vision:** off. The config has no vision model.
- **DFlash2:** Strata supports it for the Qwen3.8-Flash-Next models. On this fine-tune it accepted 28% of its drafts and
  was slower (11.0 tok/s against 18.7). Keep it off.
- **MTP:** slower on this machine (12.2 tok/s against 13.9 with TAIL_SKIP 7). Keep it off.
- **Kernel checks:** the IQ3_XXS and IQ2_S kernels pass the parity tests on a 12.8 MB copy of the model's own dense
  tensors. The full 81 GB file does not fit in one Metal buffer, so the parity tool cannot read it whole. The copy script
  is not in the repo.
- **`strata launch`:** it does not support this model. Its plan reads the file as Swift IQ3_S.

## Troubleshooting

| Message | Cause | Fix |
| --- | --- | --- |
| `mtl: no function gemv_18_16_1_4_1 ...` (also types 22, `dq_f16_18`) | No kernel for IQ3_XXS or IQ2_S | Check the five source changes in section 2, then rebuild with `build-strata-metal.sh`. |
| `native_experts.txt does not match the model` | The pack was built from another file, such as shard 1 | Rebuild the pack from the merged file (section 5). |
| `layer 13 has no routed experts` | The engine opened only shard 1 | Merge the shards (section 4). |
| `needs --spec T and --prefill CHUNK` | Missing arguments | Keep `--spec` and `--prefill` in `args`. |
| `the engine stopped unexpectedly (exit code 3)` | A kernel or memory error | Read the last `mtl:` line in `strata-orca-fastG.log`. |
| The reply is empty | All tokens went to thinking | Raise `max_tokens` to 800 or more. |
| Swap keeps growing | Memory is short | Use chunk 4096 and arena 10 GB, and close other apps. |

## Files

- `docs/ORCA_MAC_M2_README.md`: this guide.
- `strata-orca-fastG.json`: the server config (section 7). The `strata-*.json` pattern is gitignored.
- `packs/orca-iq3_xxs-merged/`: the pack (section 5). Gitignored.
- `build-metal/strata`: the engine (section 2).
- `build-strata-metal.sh`: the build script. It is not committed yet.
- `data/expert-profile.bin`: the expert profile used by the config.
- `docs/ORCA.md`: the general Orca notes.
