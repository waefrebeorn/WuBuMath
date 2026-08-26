# WUBU 50-GROUP SOTA DOMINANCE ROADMAP
# ====================================
# 153 gaps closed (existing) + ~1000 new gaps organized into 25 groups
# Each group = one focused engineering sprint
#
# TIER 1: VIDEO CODEC CORE — beat x264 at its own game
# TIER 2: ENTROPY CODING — the compression engine
# TIER 3: TRANSFORM & QUANTIZATION — where the bits are saved
# TIER 4: PREDICTION — temporal + spatial + intra
# TIER 5: FILTERING & POST-PROCESSING — quality recovery
# TIER 6: RATE CONTROL & RDO — optimal decisions
# TIER 7: COLOR & SAMPLING — perceptual efficiency
# TIER 8: CONTAINER & STREAMING — delivery
# TIER 9: HYPERBOLIC ML — our unique advantage
# TIER 10: APPLICATIONS — real-world tools

## GROUP 1: Sub-pixel Motion Estimation Foundation
- G1.01: Half-pel interpolation filter (6-tap [1,-5,20,20,-5,1]/32)
- G1.02: Quarter-pel bilinear from half-pel grid
- G1.03: Eighth-pel refinement for VVC precision
- G1.04: Multi-reference frame buffer management
- G1.05: Skip mode detection (zero residual → skip flag)
- G1.06: Merge mode for spatial MV prediction
- G1.07: Temporal MV prediction from collocated block
- G1.08: Spatial MV prediction (median of left/above/above-right)
- G1.09: Sub-block ME partitions (4x4, 8x4, 4x8)
- G1.10: Asymmetric motion partitions (2N×U, 2N×D, nL×2N, nR×2N)
- G1.11: Weighted prediction for P/B frame blending
- G1.12: Bi-prediction (average of two MV predictions)
- G1.13: Motion vector rounding to sub-pel precision
- G1.14: Search range adaptation from neighboring blocks
- G1.15: Early termination threshold based on block variance
- G1.16: Diamond search pattern optimization
- G1.17: Hexagon search pattern for larger ranges
- G1.18: UMHexagonS (uneven multi-hexagon grid search)
- G1.19: SIMD AVX2 SAD/SATD for sub-pel positions
- G1.20: Rate-distortion cost per candidate MV

## GROUP 2: Advanced Motion Compensation
- G2.01: Interpolated reference frame generation (full pel→sub-pel)
- G2.02: Weighted sample prediction process
- G2.03: Overlapped block motion compensation (OBMC)
- G2.04: Derivation process for collocated MVs
- G2.05: Long-term reference frame management
- G2.06: Reference marking and sliding window
- G2.07: Adaptive motion vector range
- G2.08: MV prediction from multiple candidates
- G2.09: Combined bi-predictive with weighted average
- G2.10: Decoded picture buffer (DPB) management

## GROUP 3: CABAC Context Models — Init
- G3.01: Probability state machine (64 states, LPS/MPS)
- G3.02: State transition tables (64×2 for MPS/LPS)
- G3.03: Range subdivision (quarter/half/three-quarter)
- G3.04: Renormalization engine
- G3.05: Initial state table initialization from QP
- G3.06: Context storage array (per-context-type)

## GROUP 4: CABAC Encoding Engine
- G4.01: Encode decision (MPS vs LPS path)
- G4.02: Encode bypass (equiprobable binary)
- G4.03: Encode terminate (end of slice)
- G4.04: Bit stuffing and byte alignment
- G4.05: Output bitstream accumulation
- G4.06: Put-bit + put-final functions
- G4.07: Range-low tracking with carry propagation
- G4.08: Pre-context init for each slice start

## GROUP 5: CABAC Decoding Engine
- D5.01: Decode decision from range subdivision
- D5.02: Decode bypass (read bit directly)
- D5.03: Decode terminate detection
- D5.04: Renormalization in decoder
- D5.05: Read-bits function from bitstream
- D5.06: Initialization from bitstream header

## GROUP 6: CABAC Coefficient Coding
- G6.01: Significant coefficient flag coding
- G6.02: Greater-than-one flag coding
- G6.03: Greater-than-two flag coding
- G6.04: Level value coding (remaining magnitude)
- G6.05: Last significant coefficient position
- G6.06: Coded block flag (CBF) coding
- G6.07: Context increment derivation
- G6.08: Neighbor-based context selection
- G6.09: Scan position dependent contexts
- G6.10: Sign coding (bypass)
- G6.11: Significant coefficient group (CG) flags
- G6.12: Multiple decimation for level thresholds

## GROUP 7: Variable-Size Transforms
- G7.01: 4×4 integer DCT (H.264 core transform)
- G7.02: 16×16 integer DCT (large blocks)
- G7.03: 32×32 integer DCT (HEVC/VVC large CTUs)
- G7.04: 4×4 DST-VII for intra residuals
- G7.05: Transform skip mode (identity transform)
- G7.06: Cross-component transform (RCT/Joint-CbCr)
- G7.07: Adaptive transform selection by RDO
- G7.08: Implicit rounding error compensation
- G7.09: Quantization matrices per size/type
- G7.10: Scaling list support

## GROUP 8: Rate-Distortion Optimized Quantization
- G8.01: RDOQ for significant flag decision
- G8.02: RDOQ for level value selection
- G8.03: RDOQ for CG-level decisions
- G8.04: Lambda computation from QP (λ=α·2^(QP/6))
- G8.05: Per-block lambda adjustment
- G8.06: Trellis quantization (Viterbi through states)
- G8.07: Dead-zone quantization tuning
- G8.08: Per-coefficient scaling lists
- G8.09: Chroma QP offset tables
- G8.10: Per-frame adaptive QP

## GROUP 9: Angular Intra Prediction (35 modes)
- G9.01: Mode table (26 angular + planar + DC)
- G9.02: Reference pixel availability and substitution
- G9.03: Angular prediction for modes 2-33
- G9.04: Displacement angle table
- G9.05: Boundary smoothing filters
- G9.06: Planar prediction implementation
- G9.07: DC prediction with edge filtering
- G9.08: Mode-dependent reference smoothing (MRLS)
- G9.09: Most-probable-mode signaling

## GROUP 10: Advanced Intra Tools
- G10.01: Multiple reference line (MRL) prediction
- G10.02: Intra sub-partitioning (ISP)
- G10.03: Matrix-weighted intra prediction (MIP)
- G10.04: Position-dependent prediction combination (PDPC)
- G10.05: Intra block copy (IBC) for screen content
- G10.06: Palette mode for screen content
- G10.07: Intra template matching prediction (TMP)
- G10.08: Cross-component linear model (CCLM) chroma prediction
- G10.09: Joint chroma residual (JCCR)
- G10.10: Local dual tree for chroma

## GROUP 11: SAO (Sample Adaptive Offset)
- G11.01: Band offset classification
- G11.02: Edge offset (4 directions)
- G11.03: Offset type and merge left/up
- G11.04: SAO parameter estimation
- G11.05: SAO reconstruction filter
- G11.06: SAO merge flags
- G11.07: CTU-level SAO decision
- G11.08: SAO rate-distortion cost
- G11.09: SAO syntax element writing
- G11.10: SAO disable flag

## GROUP 12: Adaptive Loop Filter (ALF)
- G12.01: Wiener filter coefficient derivation
- G12.02: Filter classification (variance/activity based)
- G12.03: 7×7 diamond filter shape for luma
- G12.04: 5×5 diamond filter for chroma
- G12.05: ALF on/off control per CTU
- G12.06: Multiple ALF sets and switching
- G12.07: ALF virtual boundary handling
- G12.08: CCALF (cross-component ALF)
- G12.09: ALF coefficient delta coding
- G12.10: ALF rate-distortion optimization

## GROUP 13: Film Grain Synthesis + NN Post-filter
- G13.01: Film grain parameter estimation
- G13.02: Grain synthesis lookup tables
- G13.03: Scaling and blending functions
- G13.04: Neural network post-filter architecture
- G13.05: CNN training data preparation
- G13.06: On-device inference integration

## GROUP 14: Full RDO Framework
- G14.01: RD cost computation (SSE + λ·bits)
- G14.02: Try all intra modes with full RD
- G14.03: Try all inter modes with full RD
- G14.04: Best mode decision framework
- G14.05: Split/no-split decision for CU partitioning
- G14.06: Quad-tree partitioning (QTMT)
- G14.07: Binary/ternary split modes
- G14.08: Lambda adaptation per frame type
- G14.09: Fast algorithm pruning (early termination)
- G14.10: Lookahead analysis buffer

## GROUP 15: Hierarchical B-Frame Structure
- G15.01: B-frame bidirectional prediction
- G15.02: Hierarchical GOP structure (GOP-8, GOP-16)
- G15.03: Temporal layer assignment
- G15.04: Reference picture set (RPS) management
- G15.05: Weighted prediction parameters
- G15.06: B-frame mode decision (list0/list1/bi)
- G15.07: Modified MV prediction for B-frames
- G15.08: Hierarchical QP offset assignment
- G15.09: Display order vs coding order management
- G15.10: Random access point handling

## GROUP 16: Scene Analysis & Adaptive Encoding
- G16.01: Scene change detection (histogram diff)
- G16.02: Content complexity measurement (spatial/temporal)
- G16.03: Adaptive QP from complexity
- G16.04: Adaptive resolution switching
- G16.05: Per-scene encoding parameter sets
- G16.06: Two-pass encoding infrastructure
- G16.07: VBR/CBR/CRF rate control modes
- G16.08: Buffer management (HRD model)
- G16.09: Sliding window bitrate enforcement
- G16.10: Quality smoothing across scenes

## GROUP 17: Color Space Pipeline
- G17.01: BT.601 ↔ BT.709 conversion
- G17.02: BT.2020 for HDR/WCG
- G17.03: ICtCp perceptual color space
- G17.04: Chroma format conversion (4:4:4↔4:2:2↔4:2:0)
- G17.05: Color gamut mapping
- G17.06: Tone mapping operator (HDR→SDR)
- G17.07: PQ/HLG transfer functions
- G17.08: ICC profile embedding
- G17.09: Color metadata SEI messages
- G17.10: Wide gamut residual coding

## GROUP 18: Container & Streaming Infrastructure
- G18.01: MP4 box structure writer (ftyp/moov/moof/mdat)
- G18.02: Fragmented MP4 (fMP4) segmenter
- G18.03: HLS playlist generator (.m3u8)
- G18.04: DASH manifest generator (.mpd)
- G18.05: SEI message insertion (user data, timing)
- G18.06: HRD signaling for buffer verification
- G18.07: Thumbnail/keyframe index generation
- G18.08: Bitstream extraction tools
- G18.09: NAL unit parsing/writing layer
- G18.10: Annex B ↔ MP4 format conversion

## GROUP 19: Hyperbolic Neural Network Layers
- G19.01: Hyperbolic neural ODE layer
- G19.02: Geodesic convolution on product manifolds
- G19.03: Hyperbolic attention with learnable curvature
- G19.04: Mixed-curvature autoencoder (product spaces H^n × S^m)
- G19.05: Hyperbolic GAN discriminator
- G19.06: Hyperbolic GAN generator
- G19.07: Gyrovector-space backpropagation
- G19.08: Riemannian batch normalization
- G19.09: Hyperbolic dropout (tangent space noise)
- G19.10: Curvature-aware Adam (adaptive c per layer)
- G19.11: Hyperbolic word embeddings with hierarchy preservation
- G19.12: Hyperbolic graph neural network (HGNN)
- G19.13: Hyperbolic reinforcement learning policy networks
- G19.14: Lorentz model operations (hyperboloid)
- G19.15: Product manifold operations (H×E, H×S, H×H)

## GROUP 20: Video Quality Metrics
- G20.01: PSNR computation (Y, U, V separately)
- G20.02: SSIM implementation
- G20.03: MS-SSIM multi-scale version
- G20.04: VMAF integration or simplified VMAF
- G20.05: Butteraugli (JXL's perceptual metric)
- G20.06: BD-Rate calculation between two codecs
- G20.07: Per-frame quality graphs
- G20.08: Scene-level quality aggregation

## GROUP 21: Real-World Video Tools
- G21.01: Transcoding pipeline (input→decode→re-encode→mux)
- G21.02: Bitstream thumbnail extraction
- G21.03: Keyframe-only fast decode
- G21.04: Segment extraction without re-encoding
- G21.05: Stream concatenation
- G21.06: Resolution scaling with proper filtering
- G21.07: Frame rate conversion (30↔60fps)
- G21.08: Audio muxing/demuxing
- G21.09: Subtitle track support
- G21.10: Metadata editing

## GROUP 22: Screen Content Coding
- G22.01: Intra block copy (IBC) search
- G22.02: Palette mode encoding/decoding
- G22.03: String matching (SMVD)
- G22.04: Text pattern detection
- G22.05: Sharp transformation mode
- G22.06: Hash-based IBC fast search
- G22.07: Color table compression
- G22.08: Lossless screen content mode
- G22.09: Mixed lossless/lossy regions
- G22.10: Scroll detection and coding

## GROUP 23: 360° Video & VR
- G23.01: Equirectangular projection support
- G23.02: Cubemap projection
- G23.03: Viewport-adaptive streaming
- G23.04: Spherical rotation-aware ME
- G23.05: Padding for wrap-around boundaries
- G23.06: Face packing/unpacking for cubemap
- G23.07: Quality metrics for 360° content

## GROUP 24: HDR & Wide Color Gamut
- G24.01: PQ (SMPTE ST 2084) transfer function
- G24.02: HLG (ARIB STD-B67) transfer function
- G24.03: 10-bit/12-bit pipeline
- G24.04: BT.2020 primary handling
- G24.05: MaxCLL/MaxFALL metadata
- G24.06: Luminance-preserving chroma quantization
- G24.07: HDR→SDR tone mapping curves
- G24.08: Dynamic metadata (SMPTE ST 2094)
- G24.09: HDR quality metrics
- G24.10: Backward-compatible SDR derivation

## GROUP 25: Neural Codec Integration
- G25.01: Learned end-to-end image codec (Hyperprior)
- G25.02: Neural video codec (DCVC-style conditional coding)
- G25.03: Entropy model neural network
- G25.04: Latent representation learning
- G25.05: Arithmetic coding with neural probability estimates
- G25.06: Perceptual loss function (LPIPS-based)
- G25.07: Generator-discriminator trade-off (R-D-P three-way)
- G25.08: Content-adaptive model update
- G25.09: Variable-rate neural codec
- G25.10: Hybrid traditional+neural pipeline
