# Strata on an M2 Max, 32 GB: build, measurements, commands

Machine: Apple M2 Max, 32 GB unified memory, macOS 27.2, internal SSD. Model files: Qwen3.8-Flash-Next packs from
ISTA-DASLab (Q2_0, IQ3_XXS). The model does not fit in RAM, so experts are streamed from the SSD for each token.

Every number here is from one machine, usually one run per setting (300 greedy tokens, 3 prompts: a story, an explanation,
a Python function). Expect about ±10% between runs. Where a result is invalid or unmeasured it says so.

## Build

Requires Apple Silicon, Xcode Command Line Tools, cmake and ninja. Intel builds of cmake/ninja under Rosetta build for
x86_64 and fail, so force arm64.

```sh
git clone https://github.com/truongmanhsang/Strata && cd Strata
git checkout metal-tail-skip
# pinned sources (llama.cpp's ggml and Apple's metal-cpp headers), as setup does
python3 -c "import sys; sys.path[:0]=['.','metal']; import setup as S, setup_mac as M; print(S.get_llama_cpp()); print(M.metal_cpp())"
arch -arm64 cmake -S metal -B build-metal -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DMETAL_CPP_DIR="$PWD/third_party/metal-cpp" -DSTRATA_GGML_DIR="$PWD/third_party/llama.cpp"
ninja -C build-metal strata
```

Installing the new build as an engine package (so `strata launch` uses it):

```sh
python3 packaging/package_engine.py --src <folder with strata, strata-vision, metal/kernels, metal/engine-msl, metal/ggml-common.h, metal/data> \
  --backend metal --version 0.1.56 --variant m2fast --arch apple-gpu --platform macos-arm64 --tree \
  --out ~/.local/share/strata/components/engine
strata components use 0.1.56-macos-arm64-metal-m2fast
```

Return to the stock engine at any time with `strata components use 0.1.56-macos-arm64-metal`.

## Switches that were measured to help (opt-in, all off by default)

| Switch | What it does | Effect measured here |
|---|---|---|
| `STRATA_METAL_TAIL_SKIP=R` | skip an expert read from the SSD when every token ranks it R or lower | `=7`: +10–30% decode; `=5`: a further 5–10%, answers differ more |
| `STRATA_METAL_MOE_TILES=8,1,4,4,2,4` | expert kernel tiles for M2 | ~+10% on stories (Q2_0 only, measured) |
| `STRATA_METAL_GEMV_TILES=m2` | dense GEMV tiles for M2 | +4% |
| `STRATA_METAL_F16=1` + `STRATA_METAL_F16_STREAM=1` | fp16 dense GEMVs, also while streaming | +2–9% without drafting, ~8% with DFlash2 |
| `STRATA_METAL_Q3KR=1` | Q3_K matrices re-laid for faster GEMVs (bitwise same output) | GEMVs 1.4–1.8x faster, but ~0–3% in real decode: the SSD wait sets the pace |
| `--drafting dflash2` | DFlash2 speculative drafter | code +30–35%; prose +5–10% |

## Benchmark results (tok/s, decode)

**Q2_0, 128K, arena 10 GB** (story / explanation / code):

| Step | tok/s |
|---|---|
| installed 0.1.56, default settings | ~15 |
| + arena 10 GB, prompt chunk 4096 | ~18–19 |
| + TAIL_SKIP=7 | ~20–22 |
| + expert and GEMV tiles | ~22–24 |
| + fp16 while streaming | ~23–25 |
| + DFlash2 | story 22–26, explanation 22–26, **code 26–29** |

Other Q2_0 results: arena 14 GB at 256K gave 22–23 tok/s but dropped free RAM to 7% with about 5 GB swapped: not safe.
Prompt chunk 4096 read 7.3K tokens fastest (266 tok/s). Read-ahead depth, I/O thread count, read split, file-cache
mode and layer pre-opening did not help. Stock 0.1.56 and the new build give identical greedy tokens with no switch set.

**IQ3_XXS, 256K, arena 10 GB** (one run each):

| Drafting | story | explanation | code | lowest free RAM |
|---|---|---|---|---|
| off | 18.1 | 16.8 | 16.0 | 12% |
| DFlash2 (spec 6, conf 0.8) | 19.3 | 20.1 | 20.7 | 11% |

DFlash2 accepted 200 of 211 drafted tokens on code. Both runs swapped out 100–200 thousand pages.

**Not valid:** a comparison of DFlash2 `--spec` 4 vs 6 and `--draft-conf` 0.65 vs 0.8 on IQ3_XXS. `strata launch --set spec=…`
does not reach the engine, so both runs used spec 6 / conf 0.8 (the log shows it). Not measured.

**Not measured:** IQ3_XXS with arena 11–12 GB; IQ3_XXS with the Q2_0 tile switches (they were measured on Q2_0 only).

## Best commands

Q2_0 (fastest measured, about 26–29 tok/s on code, 22–27 on prose):

```sh
strata launch --vision off --kv q4_0 --parallel 1 --context 128k --mtp off --drafting dflash2 \
  --set STRATA_METAL_ARENA_GB=10 --set conversation-cache-mib=0 --set STRATA_METAL_PROMPT_CHUNK=4096 \
  --set STRATA_METAL_TAIL_SKIP=7 --set STRATA_METAL_MOE_TILES=8,1,4,4,2,4 --set STRATA_METAL_GEMV_TILES=m2 \
  --set STRATA_METAL_F16=1 --set STRATA_METAL_F16_STREAM=1 --set STRATA_METAL_Q3KR=1 --save-preset fast
strata launch --preset fast
```

IQ3_XXS at 256K (better answers than Q2_0, about 19–21 tok/s):

```sh
strata launch --model iq3_xxs --vision off --kv q4_0 --parallel 1 --context 256k --mtp off --drafting dflash2 \
  --set STRATA_METAL_ARENA_GB=10 --set conversation-cache-mib=0 --set STRATA_METAL_PROMPT_CHUNK=4096 \
  --set STRATA_METAL_TAIL_SKIP=7 --set STRATA_METAL_F16=1 --set STRATA_METAL_F16_STREAM=1 --set STRATA_METAL_Q3KR=1 \
  --save-preset iq3_256k
strata launch --preset iq3_256k
```

Stop: `strata launch --stop`. Close memory-heavy apps first; with a 10 GB arena the machine still swaps a little.

## Known limits

- Single machine, short runs; no quality benchmark was run on the streaming paths. The greedy-token check covers the default
  path only (identical to stock), not the switches.
- `TAIL_SKIP` and the tile switches change the arithmetic; answers differ slightly from the stock engine when they are on.
- `m2fast` packages are local and unsigned. They are not in any release.
