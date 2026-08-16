# KV-Cache Quantization — Research Survey and Direction (3–6 bit)

> Status: research write-up for the `feat/kv-quantization` branch. No implementation yet.

**Goal:** Identify the most promising low-bit KV-cache quantization direction (3–6 bit) for NInfer's
registered Qwen3.6 identities, which today store KV as `BFloat16` or `Int8Group64`
(`include/ninfer/types.h:21`, per-head D256, group-64 code + FP16 scale planes).

**Method note:** Early-2024 methods (KIVI, KVQuant, GEAR) are explicitly excluded — they are stale in
this fast-moving field. The survey is scoped to arXiv work from 2025 onward, gathered via the arXiv
export API sorted by submission date (not web-search noise).

---

## 1. The field as of mid-2026

- 4-bit KV is the conservative, serving-validated floor. The active research frontier is 3-bit and
  below; 2-bit is possible with rotation but quality is fragile (the "2-bit cliff").
- The dominant technical direction is **rotation-based coding**: apply an orthogonal transform to
  keys/values before quantization to suppress outlier channels, then scalar/codebook quantize.
  This line runs QuaRot → OSCAR → OptR (2-bit) and TurboQuant → UltraQuant (3–4 bit).
- A distinct, NInfer-specific line is **RoPE-aware bit allocation**: under RoPE, a key's
  contribution to a future logit decomposes into a position-dependent sum over 2-D frequency
  blocks, so key quantization is naturally a block-wise bit-allocation problem.
- 5–6 bit is a no-man's land for KV: no leading method targets it. FP4/FP6 are hardware *weight*
  formats (Blackwell `sm_120a` native FP4/FP6/FP8), not KV-codec methods.

## 2. Candidate methods (2025–2026)

### TurboQuant — rotation + polar codebook (3-bit, the field anchor)
- Google Research (arXiv Apr 2025, ICLR 2026). Stack: QJL (1-bit JL correction) + PolarQuant
  (polar-coordinate transform) + TurboQuant (unification with provable rate-distortion bound).
- 3-bit KV with claimed zero accuracy loss; up to 8x attention-score compute speedup at 4 bits.
- Distortion provably within 2.7x of Shannon's rate-distortion optimum.
- Widely built upon; UltraQuant (below) shows the "lessons learned" 4-bit deployment recipe.

### Block-GTQ — RoPE-aware bit allocation (keys)
- arXiv 2606.24033 (Jun 2026). Built on TurboQuant-MSE (TQ-MSE), the scalar bit-width-control
  variant of TurboQuant. Per layer + KV head, computes a label-free energy score per RoPE
  frequency block and greedily allocates integer bit widths (e.g. 2/3-bit mixes) by marginal gain.
- Results: per-layer MAE cut 32–80% vs uniform TQ-MSE at 2–3 b/dim K-only; K2V2 NIAH 70.6→97.4
  and LongBench-EN 36.87→53.31 on Llama-3.1-8B-Instruct; wins all 367/367 layer comparisons.
- **Directly maps onto NInfer's Qwen3.6 MRoPE structure** — this is the NInfer-specific lead.

### RoPE-aligned rotations (the converse result)
- arXiv 2608.13365 (Aug 2026). Proves: for distinct frequencies, the only single-head orthogonal
  maps commuting with RoPE are per-pair rotations. Empirically per-pair rotations do **not** beat
  the full-head Hadamard at W4A4KV4; composing per-pair + Hadamard satisfies the ±0.05-PPL bar.
- Conclusion for NInfer: keep full-head Hadamard as the outlier-suppression rotation; do not
  chase RoPE-commuting per-pair rotations as a replacement.

### UltraQuant — deployment-tuned 4-bit
- arXiv 2606.20474 (Jun 2026). TurboQuant-style rotation + codebook with the practical deltas:
  asymmetric K/V treatment, Walsh-Hadamard rotation, **QJL removed** (the expensive piece),
  block-scale variants. Includes an FP4 path (FP8 queries, FP4 KV, UE8M0 group scales) for CDNA4.

### SPECTRA — spectral transform coding (past the 2-bit cliff)
- arXiv 2608.07915 (Aug 2026). Spends bits per-channel on a coordinate system derived from the
  cache's own statistics to pass the 2-bit cliff. Supports the rotation + adaptive-bit-spending
  thesis; not yet a NInfer-specific lever.

### 2-bit rotation lineage (background only)
- QuaRot (2024) → OSCAR (offline spectral covariance rotation, 2605.17757) → OptR (output-aware
  rotation minimizing post-`W_O` error, 2608.02691). These are the INT2 quality tricks.

### DGAP / local distribution restoration (evaluation criterion)
- arXiv 2607.16248 (Jun 2026). Not a codec: the failure mode of low-bit KV is structured local
  misranking in the top-K region, not absolute logit error. Restores the top-K candidate
  distribution at decode. K1V1 RULER 47.8→83.2 on Llama-3.1-8B. Any aggressive 3–4 bit scheme
  should be evaluated against top-K-ranking preservation, not just logit/perplexity.

## 3. Recommendation

**Pursue Block-GTQ (RoPE-aware bit allocation) as the leading design**, composed with TurboQuant's
rotation/transform machinery, because:

1. It is the only leading method that exploits **RoPE structure**, and NInfer's Qwen3.6 identities
   are RoPE/MRoPE models — the structure is identical and directly addressable.
2. It operates in the 2–3 bit key regime, which is exactly the frontier band the branch targets,
   and its bit-allocation knob naturally generalizes to per-block 3/4/5-bit mixes if 5–6 bit is
   ever wanted.
3. It builds on TQ-MSE, a scalar quantization path — no polar codebook/QJL required at 4-bit,
   keeping the implementation close to NInfer's existing group-code + scale-plane codec.
4. The Aug-2026 converse result tells us exactly what not to do (RoPE-commuting per-pair
   rotations), saving a known dead end.

Open design questions to resolve next (see `2026-08-16-kv-block-gtq.md` for the deep dive):
- Whether keys and values get different treatment (asymmetric K/V is the field's consensus).
- Where rotation sits relative to MRoPE (per-head Hadamard before or after the RoPE apply).
- How bit-allocation metadata is stored in the paged KV layout without material per-token overhead.

## 4. Sources

- arXiv 2606.24033 (Block-GTQ): https://arxiv.org/abs/2606.24033
- arXiv 2608.13365 (RoPE-aligned Q/K rotations): https://arxiv.org/abs/2608.13365
- arXiv 2608.07915 (SPECTRA): https://arxiv.org/abs/2608.07915
- arXiv 2606.20474 (UltraQuant): https://arxiv.org/abs/2606.20474
- arXiv 2608.04074 (attention-preserving KV VQ): https://arxiv.org/abs/2608.04074
- arXiv 2608.02691 (OptR output-aware rotation): https://arxiv.org/abs/2608.02691
- arXiv 2607.16248 (DGAP local distribution restoration): https://arxiv.org/abs/2607.16248
- arXiv 2605.17757 (OSCAR): https://arxiv.org/abs/2605.17757
- TurboQuant blog overview: https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/
- TurboQuant/vector-quantization analysis: https://shbhmrzd.github.io/ai/systems/ml-infrastructure/quantization/2026/04/04/turboquant-vector-quantization-for-llm-inference.html
