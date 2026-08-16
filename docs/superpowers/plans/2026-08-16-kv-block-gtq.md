# Block-GTQ Deep Dive — RoPE-Aware Bit Allocation for NInfer KV Quantization

> Source: arXiv 2606.24033 "RoPE-Aware Bit Allocation for KV-Cache Quantization" (Fengfeng Liang,
> Yuechen Zhang, Jiaya Jia; CUHK). Companion to `2026-08-16-kv-cache-quantization-research.md`.
> This is a research assessment, not an implementation plan.

---

## 1. What Block-GTQ is

Block-GTQ reframes low-bit **key**-cache compression for RoPE models as a **block-wise rate
allocation** problem. It keeps every RoPE block stored, but spends a different integer bit width on
each 2-D RoPE frequency block inside a KV head, instead of one uniform width.

**Why it works:** under RoPE, a query-key logit decomposes exactly over 2-D frequency blocks:

```
K∆(q,k) = qᵀ R∆ k = Σ_i q(i)ᵀ R(∆θi) k(i)
```

with each block an orthogonal 2×2 rotation. The induced logit error from key quantization is
likewise block-wise independent (Cauchy–Schwarz, no cross-block terms). So the right allocation is:
give more bits to blocks whose energy `E[‖q(i)‖·‖k(i)‖]` is high. RoPE preserves per-block ℓ2 norms,
so the energy score can be computed in pre-RoPE coordinates.

**Local encoder — TQ-MSE:** Block-GTQ reuses TurboQuant-MSE as its local quantizer. TQ-MSE, for a
nonzero vector x: normalize → apply a shared orthogonal rotation (Walsh-Hadamard) → scalar-quantize
the rotated coordinates → restore the radius (stored per-vector norm). Its rate law is
`E‖x−x̂‖² ≤ ‖x‖² C_TQ 4^(−b)` with `C_TQ = 3π/2`: each extra bit quarters the MSE bound. TQ-MSE is
TurboQuant minus the polar codebook/QJL — a scalar path, close to NInfer's existing group-code
codec.

## 2. The allocator (Algorithm 1, Theorem 1)

- **Energy score** (label-free, from marginal Q/K second moments, per layer ℓ and KV head h):

  ```
  s(ℓ,h,i) = ½ [ E_{t,g∈G(h)} ‖q(i)_{ℓ,g,t}‖² + E_t ‖k(i)_{ℓ,h,t}‖² ]
  ```

  averaged over a short unlabeled calibration prefix. `G(h)` = the query heads reading KV head h.
  By AM-GM this overestimates but never underestimates the true block weight `E[‖q(i)‖·‖k(i)‖]`.

- **Objective:** for a head-level integer budget B (bits), pick `b = (b1..bL)`, `bmin ≤ bi ≤ bmax`,
  `Σbi = B`, minimizing `J(b) = Σ_i s_i · 4^(−bi)`.

- **Greedy is provably optimal** for this objective (Theorem 1): initialize all blocks at `bmin`,
  repeatedly assign the next bit to the block with the largest marginal reduction
  `Δ_i(bi) = (3/4)·s_i·4^(−bi)`. A max-priority queue implements this; each bit received quarters
  the block's next marginal gain.

- **Physical layout:** blocks with the same assigned width are concatenated into one rate group and
  encoded by one TQ-MSE encoder at that width. Uniform TQ-MSE is the all-blocks-same-rate special
  case. Values don't enter the RoPE logit, so V gets uniform TQ-MSE.

## 3. Packed serving path (the deployment recipe)

- Cache holds packed K/V code streams + norms + static layout metadata in HBM. **No fp16 KV is ever
  materialized in HBM.**
- The fused attention kernel loads only the current key tile, unpacks each segment (nibbles for
  ≤4-bit groups, bytes for higher-bit groups), dequantizes through a shared fp16 codebook that fits
  in L1, rescales by per-group K norms and per-token V norms, then runs QKᵀ and PV as fp16-input /
  fp32-accumulate tensor-core matmuls under a fully fp32 online softmax.
- Decode is memory-bandwidth bound (one query reads all T keys), so per-step HBM traffic is only
  packed codes + norms: **~157 B/token/KV-head at K3V3 vs 512 B fp16 (≈3.26×)**.
- Prefill populates the cache per layer in full-length passes (two batched kernels, one for K one for
  V), then a FlashAttention-2-style kernel streams compressed tiles.
- The Q-side rotation is folded into the q-projection weights offline (a small QRᵀ matmul).
- Cache update captured as a CUDA graph to remove per-step launch overhead.

## 4. Results (as reported)

- Ten-model GQA panel incl. Qwen3-8B, Qwen3-30B-A3B; 2 and 3 b/dim K-only. Per-layer RoPE-logit
  MAE cut **32–80%** vs uniform TQ-MSE; **367/367** layer wins at both budgets.
- K2V2 on Llama-3.1-8B-Instruct: NIAH 70.6→97.4, LongBench-EN 36.87→53.31.
- AIME 2024/25, no fp16 recent-key buffer, K3V2 on DeepSeek-R1-Distill-Qwen-7B: 51.7/37.5 vs fp16
  54.2/37.9 — uniform TQ-MSE collapses to 0.0/0.0. (On the Llama-8B backbone it trails PM-KVQ, a
  joint K+V loss-gradient allocator; the authors flag V-side allocation as future work.)
- Deployment (single H800, Qwen2.5-3B-Instruct, K3V3): 3.24× KV compression, fp16-comparable PPL,
  1.34× faster than fp16 FA-2 at 128K, peak memory 56.31 GB→19.85 GB, feasible at 256K/512K where
  fp16 OOMs. Uniform TQ-MSE decodes ~14% faster but collapses in quality.
- Calibration is robust at K3V3 (6 NIAH subtasks within 1.07 pp, PPL within ±1σ across seeds and
  corpora); K2V2 is more sensitive because a misplaced bit at b=3 costs 4× less than at b=2.

## 5. Fit against NInfer's Qwen3.6 geometry

**Direct structural matches:**
- Registered full-attention geometry is `[256, 24|4, W, B]` group 6 (27B) and `[256, 16|2, W, B]`
  group 8 (35B-A3B) — D256 per KV head, exactly the block decomposition Block-GTQ targets
  (L = 128 2-D blocks per head).
- KV storage is already paged, per-head, post-RoPE (append happens after `partial_mrope`), with the
  existing `Int8Group64` codec sharing the same code+scale-plane shape TQ-MSE uses.
- Both registered targets are in the paper's validated backbone family (Qwen GQA / MRoPE).
- Per-step bandwidth-bound decode is precisely the regime where packed low-bit KV pays off.

**NInfer-specific differences to resolve:**

1. **Partial MRoPE.** Only 64 of 256 dims per head are rotated (`rotary_dims=64`), split across
   interleaved MRoPE sections `[11,11,10]` (32 rotated 2-D blocks; θ_text = 1e7, θ_vision = 1e4).
   The remaining 96 unrotated 2-D blocks have identity rotation. The logit still decomposes
   block-wise (unrotated blocks contribute `q(i)ᵀk(i)`), so the allocator applies to all 128 blocks
   — but the energy-score and rate law must be checked against a partial-RoPE head, and the
   "position-dependent" argument only drives the 32 rotated blocks.
2. **MRoPE 3-axis positions.** Text-only decode uses one scalar position for all sections (matches
   the paper's single-∆ analysis). Multimodal prefill supplies distinct temporal/height/width
   positions, so a block's effective rotation depends on which axis owns it; the block-wise bound
   still holds, but calibration must average over the actual multimodal position mix.
3. **Bit-budget range vs the branch goal.** Block-GTQ's operating points are K2V2/K3V2/K3V3 —
   **2–3 bit**, i.e. the bottom of the branch's 3–6 bit band. Its allocator generalizes upward
   (raise the head budget B and/or `bmax` for 4–5 bit mixes), but the paper does not evaluate those.
4. **Calibration requirement.** Energy scores need a short labeled-free prefix per layer/KV head.
   NInfer's startup-fixed architecture must pick where calibration runs (a fixed startup sequence vs
   a first-request observation) and how scores are stored (they are model+head constants, not
   per-token).
5. **Decode crossover.** The paper shows packed decode is *slower* than fp16 at short context (T≲64K)
   because of in-kernel unpack overhead, overtaking only when KV bandwidth dominates. NInfer must
   measure the crossover against its current INT8-G64 decode path at its real context mix, not just
   peak capacity.
6. **MTP / DFlash caches.** The 35B-A3B DFlash KV is a cyclic fixed window, and both targets have a
   one-layer MTP KV cache. Block-GTQ's allocator applies per cache; the DFlash window's small size
   may not repay per-head metadata.
7. **V-side allocation.** Block-GTQ leaves V uniform; the paper itself flags V allocation (as in
   PM-KVQ) as the natural improvement. NInfer should treat "uniform V + Block-GTQ K" as the baseline
   and evaluate a V allocator separately.

## 6. Why this over the alternatives

- **TurboQuant full stack** (polar codebook + QJL): stronger asymptotic guarantees but a
  fundamentally different decode path (polar reconstruction) than NInfer's scalar group codec.
  TQ-MSE keeps the scalar path.
- **Uniform scalar 4-bit** (e.g. UltraQuant-style): simpler, but throws away the free RoPE-block
  information that Block-GTQ exploits and cannot reach 3-bit quality.
- **Per-pair RoPE-commuting rotations** (arXiv 2608.13365): explicitly shown *not* to beat the
  full-head Hadamard — a known dead end, avoided.

Block-GTQ is the only leading method whose core idea is a direct property of the Qwen3.6 model
(RoPE/MRoPE block structure), operates in the 3-bit frontier band, and composes with the existing
group-code + scale-plane KV architecture.

## 7. Open questions for the next step

1. Where calibration runs at startup and how scores persist in the load path.
2. Layout of the mixed-rate packed K stream inside the existing paged KV plane (nibble/byte segments
   + per-group norm + schedule metadata), and whether per-head `[128]`-entry schedule tables stay
   within the current page-bytes accounting.
3. Whether the Walsh-Hadamard Q-side rotation can be folded into the registered Q4 `query_key`
   projection weights offline, or must run per-token.
4. MTP and DFlash cache scope.
5. Verification criterion: top-K ranking preservation (DGAP-style) plus RoPE-logit MAE, not just PPL.

## 8. Sources

- Paper PDF: https://arxiv.org/pdf/2606.24033v1
- Code (Triton, MIT): https://github.com/JIA-Lab-research/blockgtq
- TurboQuant-MSE reference [41] and the TurboQuant blog:
  https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/
- Converse result on RoPE-commuting rotations: arXiv 2608.13365
