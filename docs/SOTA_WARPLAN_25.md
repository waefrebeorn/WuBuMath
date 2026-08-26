# WUBQ SOTA WAR PLAN v2 — 25 Categories
**Date:** 2026-08-26 · **Repo:** /home/wubu/wubumath (C11) · **Doctrine:** beat x264/x265 on REAL content, measured honestly, or no victory claims.

---

## THE TARGET LINE (measured 2026-08-26, real 1080p YouTube, 10s @30fps yuv420p)
| Content | x264 crf23 | x265 crf23 | VP9 crf34 |
|---|---|---|---|
| Anime | 2.29MB @ 33.60dB (408×) | 1.41MB @ 33.60dB (660×) | 1.34MB @ 33.58dB (694×) |
| Movie | 3.59MB @ 32.82dB (208×) | 2.78MB @ 32.87dB (268×) | 2.08MB @ 32.68dB (359×) |
| Variety | 3.34MB @ 34.10dB (223×) | 2.15MB @ 34.14dB (348×) | 1.80MB @ 34.08dB (414×) |

**Victory = WUBQ ≤ these bytes at ≥ these PSNR.** Intermediate checkpoint: beat
x264 crf23 first, then x265, then VP9. Sources kept in ~/codec_ab1080/ for reruns.

## THE 8K MANDATE (added same day)
8K (7680×4320) is the endgame battlefield — this is where VVC's advantage widens
most (+30% over HEVC at production settings) and where nobody has spare compute.
- **Corpus upgrade:** re-pull the 3 content types at 4320p where YouTube offers VP9/AV1 8K masters (Big Buck Bunny has a native 8K upload); cut identical 5s @ 30fps segments (8K raw frame = 124MB, so shorter segments).
- **Target line to measure:** x265 + x264 crf23 + VP9 at 8K on the corpus, same protocol. That number becomes the victory bar.
- **Compute reality:** encode times explode ~16× vs 1080p (pixels) plus worse cache behavior (~24× wall). SIMD saturation (C16), tile/WPP parallelism, and ME early-exit are no longer optional speed features — they are the enabling tech for even RUNNING the 8K benchmarks.
- **Where WUBQ can win first:** domain-specific rotational/animation content at 8K — the quaternion path's cost scales with motion complexity, not pixel count, so the ratio advantage GROWS with resolution.
- New gap items folded into categories: C5 += hierarchical ME / pyramid search (mandatory for 8K ME speed); C16 += tile-parallel encode verified at 8K; C17 += 8K container headers + level signaling; C19 += 8K corpus entries in manifest.

## GAAD RESOLUTION DOCTRINE — ALL RESOLUTIONS, ONE MATH (user directive)
No more fixed-resolution thinking. Every codec stage operates on **GAAD regions**,
not pixel grids: Recursive Golden Subdivision (square + residual golden rectangle,
recursed to block size) + Phi-Spiral Sectoring for focal flow. Reference theory:
`docs/theory/papers/GAAD-WuBu-ST1.md` / `-ST2.md`.
- **Resolution ladder:** any W×H → GAAD-subdivide until leaf regions hit the
  native transform sizes (8/16/32). Same math at 176×144 and 7680×4320; only the
  subdivision depth changes (depth ≈ log_φ(W/min_block)).
- **Block partitioning = golden split**, not binary quadtree: CU splits square |
  golden-rectangle instead of 4 quadrants; rectangles recurse again. This gives
  aspect-native partitions (no distortion of anamorphic content) and is the
  partition scheme VVC doesn't have.
- **ME scales hierarchically:** coarse GAAD levels predict motion first; fine
  levels refine within inherited vectors — mandatory for 8K ME cost.
- **Rate allocation by φ:** bits distributed across GAAD region hierarchy in φ-
  geometric proportions (focal spiral sectors get cφ² weight, periphery c).
- New category folded in: **C26-GAAD integration** — replace fixed-grid CTU path
  with golden-subdivision partitioner; exit = equal-or-better BD-rate vs quadtree
  at BOTH 1080p and 8K, and identical code path handling arbitrary W×H without
  special cases.

## 8K TARGET LINE — MEASURED 2026-08-26 (native 8K Big Buck Bunny, 3s @24fps)
Raw y4m: 3.58GB for 72 frames. Encode cost reality check on this box (5GB RAM):
x265 ultrafast took 689s (0.10 fps) — 8K encoding is compute-bound, confirming
C16 parallelism as enabling tech.
| Codec | Bytes | Ratio vs raw | PSNR |
|---|---|---|---|
| **x265 crf23** | **1.90 MB** | **1889×** | **40.09 dB** |
| VP9 crf34 (realtime) | 4.30 MB | 833× | 40.11 dB |
| x264 crf23 | 14.08 MB | 254× | 40.20 dB |

**The 8K victory bar: beat 1.90MB @ ≥40.1dB on this segment.**
Note the resolution effect: ratios are ~5× better than 1080p across the board
(more spatial redundancy per frame) — exactly where GAAD hierarchical prediction
and the quaternion path's motion-complexity scaling should compound.

## 2026 EXTERNAL STATE OF ART (researched this session)
- ECM (MPEG research model): ~28% beyond VVC; H.267 race started; NN tools now standard ingredients.
- Neural video codecs (DCVC-FM/UF, GIViC, C3): match/beat VTM but complexity-heavy; INR variants encode at milli-FPS.
- Generative/diffusion codecs: dominate ultra-low-bitrate perceptual space (CoD, GVC-RT, CADC); real-time diffusion codec hits 85% bitrate cut vs MS-ILLM-class.
- Audio: Opus still wins transport; neural codecs win as LLM tokenizers; OAC (AOMedia) = next traditional bet.
- Lossless text: CM/PAQ-family still king offline; zstd rules speed tier; xEnc3 new hybrid contender.
- Gaussian splats: MPEG exploration track for next-gen video; GSVC shows splat-video viability.
- Point cloud: V-PCC (dense) + G-PCC (sparse) split the field.

---

## THE 25 CATEGORIES

### TIER A — ENTROPY ENGINE (the bits themselves)

**C1. CABAC round-trip integrity**
- Gaps: decoder sync bug (~17/200 bin mismatch). Trace low/range/offset divergence step-by-step between encoder carry-out and decoder offset init.
- Exit: 10^7 random bins round-trip exact, both skewed and uniform distributions; ASAN clean.
- Why first: every category below pays rent through this pipe. Current 0.36 bits/bin encoder is useless until decode closes.

**C2. Context modeling beyond H.264 baseline**
- Gaps: per-context-type init from QP; dual-context (like HEVC's split flag); context derivation from neighbor statistics instead of fixed tables; adaptive context refresh per GOP.
- Exit: ≥5% bitrate drop vs flat contexts on anime+movie segments at same PSNR.

**C3. Arithmetic coder alternatives**
- Gaps: binary range coder vs multi-symbol arithmetic comparison; rANS pipeline for coefficient tokens (AV1-style multithread-friendly); verify against zlib-stage results.
- Exit: pick winner by bits/frame AND decode parallelism; document trade table.

**C4. Entropy of residuals — symbol alphabet design**
- Gaps: coefficient tokenization (run/level pairs vs zigzag varint vs Golomb family); joint coding of EOB+run; context-coded last-coefficient position.
- Exit: on identical quantized coefficients, best scheme beats current zigzag by ≥15%.

### TIER B — PREDICTION (never code what you can predict)

**C5. Motion estimation completeness**
- Have: sub-pel Wiener, diamond/hexagon, skip/merge/AMVP, DPB, OBMC, bi-pred.
- Gaps: UMHexagonS full implementation; search-range adaptation quality; sub-block ME partitions wired into RDO; affine/global motion models (VVC-style 6-param).
- Exit: BD-rate vs current ME on all three content types; anime expected biggest gain from global motion.

**C6. Intra prediction depth**
- Have: DC/vertical/horizontal/plane + angular core (35 modes).
- Gaps: MRL (multi-ref line), MIP (matrix intra pred), ISP (split sub-partition), cross-component (CCLM exists — needs integration), PDPC positional filtering, reference sample smoothing.
- Exit: screen-content + anime intra frames ≥10% smaller than current angular-only.

**C7. Temporal hierarchy & B-frame structure**
- Gaps: hierarchical B (GOP pyramid), weighted/bi-pred already exist but no pyramid scheduling; lookahead-driven mini-GOP; low-delay vs random-access profiles.
- Exit: RD curve at equal PSNR shows ≥20% saving vs flat IPPP on movie content.

**C8. Scene-adaptive encoding**
- Have: scene-change histogram detection, complexity metric, CRF mode, VBV, lookahead.
- Gaps: per-scene GOP sizing; grain-aware QP offsets; animation-specific mode (flat regions → long skips, line preservation); variety-show cut detector tuning (flash frames, hard cuts).
- Exit: each content type gets a tuned preset that beats generic preset by ≥8%.

### TIER C — TRANSFORM & QUANTIZATION

**C9. Variable-size transforms**
- Gaps: 4×4–64×64 DCT2 family complete; DST-VII/DCT-VIII for small intra blocks (VVC); transform selection signaling; large-block partitioning interplay with TU tree.
- Exit: full transform-size matrix tested on all segments; ≥12% over 8×8-fixed.

**C10. Perceptual quantization**
- Gaps: trellis RDOQ exists (matches rounding) — integrate into encoder loop; perceptual qmats (flat vs visual); adaptive quant per CU from variance + luma masking; chroma QP tables.
- Exit: SSIM-per-byte improvement ≥10% vs flat quant on movie segment.

**C11. Frequency-domain tricks**
- Gaps: zero-coefficient early exit stats; transform-skip mode for noisy/grainy blocks; rotated transforms for directional residual energy; implicit MTS selection in RDO.
- Exit: measured per-tool gain attribution table.

### TIER D — LOOP FILTERS & RECONSTRUCTION

**C12. Deblocking beyond H.264**
- Have: full H.264 deblocking.
- Gaps: VVC-style DBF (longer taps, variable boundary strength); luma/chroma separate decisions; integrated filter decision in RDO rather than post-hoc.
- Exit: PSNR delta at same bitrate vs H.264-filter baseline.

**C13. SAO + ALF**
- Have: band + edge SAO.
- Gaps: ALF (adaptive loop filter, Wiener-derived, CCALF cross-component); filter-on/off flags in bitstream; encoder-side filter parameter estimation via least squares.
- Exit: ALF+SAO stack gains ≥4% BD-rate on all three types.

**C14. Film grain synthesis & NN post-filters**
- Gaps: grain estimation → parametric model → resynthesis at decode (banding killer at low bitrate); optional lightweight CNN deblock/denoise as signaled tool.
- Exit: low-bitrate operating point (≤1Mbps-equiv) subjective + MS-SSIM improvement without byte cost explosion.

### TIER E — ARCHITECTURE OF THE ENCODER

**C15. Rate-distortion optimal decisions end-to-end**
- Have: lambda-from-QP, SSE cost, early SKIP/3-stage decision, CU depth pruning.
- Gaps: unified RDO across all tools (ME modes × intra modes × transforms × filters); proper λ-psychovisual adjustment; dependent RDO-QT (partition decision aware of children costs).
- Exit: single-pass encoder where every block takes globally-best mode; ablation proves each tool's contribution.

**C16. Parallelism & SIMD saturation**
- Have: AVX2 SAD 6.66×, thread pool, DCT butterfly batch, CPU feature detect, 3.3M fps ME benchmark.
- Gaps: wavefront parallel encoding (WPP); tile-based encode with dependency tracking; AVX-512 paths; frame-level parallelism for B-pyramid; lock-free DPB.
- Exit: ≥8× scaling on this box's cores at equal output bytes vs single-thread.

**C17. Container, streaming & delivery**
- Have: .WUBV container, NAL-ish units, MP4 ftyp, Annex B parsing.
- Gaps: seek index (keyframe directory), progressive decode ordering (Hilbert scan advantage), HDR metadata boxes, WebM/MP4 muxing for A/B delivery, CRC-per-GOP resilience.
- Exit: random access to any second <100ms; survives truncated stream decode.

### TIER F — METRICS & HONESTY INFRASTRUCTURE

**C18. Quality metrics arsenal**
- Have: PSNR, SSIM, MS-SSIM, BD-Rate.
- Gaps: VMAF integration (libvmaf), IWSSIM/FSIM, temporal metrics (flicker, judder), per-content-type calibration curves, automated Triple-DA report generation.
- Exit: every benchmark auto-produces a BD-rate plot + VMAF + adversarial checklist; no hand-made claims.

**C19. Benchmark corpus institutionalization**
- Gaps: formalize the 3-content corpus (anime/movie/variety) + Felix cartoon legacy set + JVT standard sequences (if downloadable); versioned MD5 manifest; one-command rerun script `tools/run_ab.sh`.
- Exit: any future claim reproducible by single command; corpus documented in README.

### TIER G — THE WUBU DIFFERENTIAL (our unfair advantages)

**C20. Quaternion/rotation domain codec (WUBQ core)**
- Gaps: define the domain honestly — rotation/animation content only; SLERP prediction is exact there (proven 6526×). Build auto-detection: measure per-segment rotational dominance, route blocks to WUBQ path vs conventional path.
- Exit: hybrid encoder where rotational segments get quaternion path automatically; combined ratio beats pure-conventional on animation corpus.

**C21. Hyperbolic manifold priors**
- Gaps: hyperbolic embedding of block-mode statistics (tree-like mode distributions compress better on Poincaré ball); gyromidpoint-based predictor fusion; curvature-adaptive entropy modeling.
- Exit: measurable bits saved from hyperbolic-context CABAC vs Euclidean context, ≥3%, on real footage. This is the WuBu signature — if it fails honestly, document why.

**C22. Neural wrapper tools (NMVC-style)**
- Gaps: tiny NN intra predictor (following JVET CfE evidence that 2-3 wrapped NN tools give real gains); NN-based fast mode decision replacing brute RDO; learned loop filter. All inference-only, C-callable, no training in the encoder hot path.
- Exit: each NN tool individually ablated; keep only those with ≥1.5% gain at acceptable complexity.

### TIER H — FRONTIERS (the mega-plan bets)

**C23. Generative ultra-low-bitrate mode**
- Gaps: diffusion/INR-based extreme-compression branch for ≤0.05 Mbps-equivalent operation where generative reconstruction beats pixel codecs perceptually; semantic token side-channel (audio-LM tokenizer insight applied to video).
- Exit: at ultra-low bitrate, VMAF/FID beats x265 even if PSNR loses; honest two-column report.

**C24. Splat & volumetric tracks**
- Gaps: 2D Gaussian splat video representation (GSVC-line) as alternative internal representation for layered content; point-cloud attributes via our entropy engine (G-PCC-style sparse coding reuse).
- Exit: splat path viable on one test sequence; entropy engine proven reusable outside pixels.

**C25. Audio + joint A/V rate control**
- Gaps: psychoacoustic module exists — finish Opus-class A/B harness; joint audio/video bit allocation under total-budget constraint (av_fidelity module wiring); container carries synced streams.
- Exit: 10s A/V clip encoded under fixed budget beats ffmpeg default muxed equivalent at same perceived quality.

---

## EXECUTION ORDER (dependency truth)
```
Wave 1: C1 (CABAC sync) ── nothing else matters without it
        C19 (corpus+script) ── honesty infrastructure in parallel
Wave 2: C9 C10 (transform+quant) C4 (symbols) ── feeds the fixed CABAC
        C5 C6 (prediction depth)
Wave 3: C13 C12 C14 (filters) C7 (B-pyramid) C15 (unified RDO)
Wave 4: C2 C3 (context+coder choice) C16 (parallel) C17 (container)
Wave 5: C20 C21 (differential paths) C22 (NN wrappers)
Wave 6: C23 C24 C25 (frontier bets) C11 C18 closeout
```

## HARD RULES (carried forward)
1. Real content or it didn't happen.
2. Same input for every codec; PSNR/VMAF from decoded output.
3. No victory claims without Triple Devil's Advocate + rerunnable command.
4. No videos/presentations until SOTA beaten on the corpus above.
5. Every gap closed = test green + register updated + commit both repos + parity gate.
