Title: Running Qwen3.8-Flash-Next (125B MoE) on a 32 GB M2 Max at 20-29 tok/s with a local Metal engine

I'm running a 2-bit and 3-bit quant of Qwen3.8-Flash-Next (125B total, 6B active, 512 experts) on a 32 GB M2 Max. The
model is 76 GB, so most experts stream from the SSD for every token. Starting point was about 15 tok/s.

What got it faster, in order of impact:
- a bigger expert cache in RAM (10 GB instead of under 1 GB when RAM was tight)
- skipping SSD reads for experts that every token ranks in its lowest three choices (about 10-30%, answers change slightly)
- DFlash2 speculative drafting (code +30-35%, prose +5-10%)
- M2-specific kernel tiles and fp16 dense GEMVs while streaming (a few percent each)

Results (single machine, one run per setting, 300 greedy tokens):
- Q2_0 at 128K: about 26-29 tok/s on code, 22-27 on prose, with everything above on
- IQ3_XXS at 256K: about 19-21 tok/s with drafting, 16-18 without

Things that did not help: read-ahead depth, I/O thread counts, file-cache mode, and a re-laid Q3_K format. The last one was
1.4-1.8x faster per kernel but made no visible difference, because the SSD wait sets the pace.

Caveats: these are my switches on a branch of my own fork, not an official release. Nothing is signed. Quality beyond reading
the answers was not measured for the streaming switches. The fork is at github.com/truongmanhsang/Strata (branch
metal-tail-skip), and the write-up with build steps and exact commands is in docs/M2_MAX_32GB.md.

Happy to answer questions about the build or the measurements.
