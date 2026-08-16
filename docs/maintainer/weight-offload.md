# CPU/system-RAM weight offload design

Status: implementation in progress. Surfaces 1-6 of the change-surface table are implemented
(see section 5): artifact host placement, tensor host addresses, the 27B residency profile, the
staging arena/slot binding, the graph-capture staging interleave, and the public engine residency
option with load/memory-summary reporting. A partial-residency knob
(`EngineOptions::resident_ffn_layers`, CLI `--n-ffn-layers N`) keeps the first N FFN text layers
device-resident while the rest stream. The remaining test/documentation surfaces (7-8) are
not yet implemented. Scope target is the Qwen3.6-27B dense identities (`groupwise-int` and
`nvfp4`); the 35B-A3B MoE target is a bonus item with extra work.

## 1. Problem

All registered identities have weight payloads larger than 16 GiB, so none can run on a 16 GB
`sm_120a` GPU (for example RTX 5060 Ti 16 GB) under the current single-device-arena load path.
The weight payload of `qwen3.6-27b/groupwise-int` is 18,323,855,104 bytes (~17.07 GiB,
[qwen3.6-27b-artifact.md](qwen3.6-27b-artifact.md) section 13); the NVFP4 artifact is slightly
larger (1301 tensor objects vs 1118). Reducing KV or context cannot make any identity fit, because
the weights alone exceed the card. Supporting these GPUs requires offloading (a subset of) the
weights to CPU/system RAM.

## 2. Current memory model (what offload must integrate with)

- `src/artifact/materializer.cpp` allocates one `DeviceArena` with capacity
  `plan.device_capacity_bytes` (the whole weight set) and streams the `.ninfer` payload into it.
- All weight `Tensor`/`Weight` views (`materialized_weight`, `src/artifact/typed_binding.cpp`) are
  plain device pointers into that arena. The NVFP4 linear builds `cuTensorMapEncodeTiled` maps from
  those addresses (`src/ops/linear/nvfp4/nvfp4_w4a4_tma.cuh:43`).
- Decode runs captured graphs: `ProgramImplCore::prepare_graphs()`
  (`src/targets/qwen3_6/impl/runtime/program_impl.h:1007`) captures one graph per
  (execution-frontier profile, batch size); each graph contains the whole 64-layer decode pass.
  Kernels and TMA maps are bound to arena addresses at capture and must stay stable on replay.
- KV (`src/runtime/engine/kv_capacity.cpp`), workspace
  (`src/targets/qwen3_6/impl/runtime/workspace_recipe.h`), and `RequestMemory` are separate frozen
  device allocations, sized from whatever device bytes remain after the weights.
- The binder's `retain_on_host` is resource-only; there is no host-resident weight path.

## 3. Chosen mechanism and why

The design follows llama.cpp's proven model (commit `1692f9e50`): weights are host-resident,
GPU computes everything, and weights stream H2D into a small resident double-buffered compute area
at compute time (`ggml_backend_sched_compute_splits`, `cur_copy[0/1]`, overlapping copy N+1 with
compute N). Kernels are unchanged; they always read device memory.

CUDA constraints were validated against the Programming Guide 13.3, PTX ISA 9.3, and Driver API
13.3 (toolchain is CUDA 13.1):

- Linux HMM (kernel 6.1.24+/6.2.11+/6.3+, CC 7.5+, driver 535+ open modules) supports full unified
  memory with software coherence; kernels fault-and-migrate managed pages natively. This is the
  only part of the managed-memory story that is documented as supported.
- TMA cannot legally target host/system memory: bulk-copy and TMA source/destination tables and
  the PTX syntax enumerate only `global` and `shared::*`. NInfer's NVFP4 linear is TMA-based, so
  offloaded weights must be copied into resident global memory before kernels and TMA touch them.
- CUDA Graph capture/replay with page faults on non-resident managed memory is undocumented; there
  is no prefetch node type, and nothing blesses faulting during replay. H2D `cudaMemcpyAsync`,
  however, is a supported capturable memory-copy node.

Consequence: managed-memory/UVM is rejected. For a dense model every page migrates on first touch
and stays (no eviction control), so the whole ~17 GiB would try to become device-resident, and TMA
from host pages is unsupported. The sound path is an explicit pinned-host weight store plus
resident staging slots with fixed addresses, fed by H2D copy nodes interleaved into the existing
captured graphs.

## 4. Design

### 4.1 Host weight store

A second materialization target keeps offloaded tensor payloads in a host store (mmap'd file or
`malloc`/`cudaHostAlloc`). The store must be pinned (`cudaHostRegister`) so H2D copies use real DMA
and the copy nodes overlap kernel compute. The `.ninfer` reader already streams payloads through
pinned staging; writing to a host store instead of the device arena is a small change.

### 4.2 Per-tensor residency profile (FFN-first offload)

Offload granularity is per tensor, not per layer. The primary offload unit for 27B is the **FFN
(SwiGLU MLP) matrices** of each layer: the fused gate/up projection and the down projection. They
are the dominant bytes in every layer (roughly 60% of layer weight bytes; the largest single object
is the MLP gate/up at ~95 MB at Q4 groupwise). Keeping the attention and GDN projections, norms,
the small conv/recurrent state, token embedding, output head, and the MTP/draft/vision extras
resident leaves the entire attention path on device across all 64 layers while only FFN matrices
stream.

The residency profile is a startup knob: a per-tensor device-vs-host decision plus a resident-bytes
budget. A default "FFN-offload" profile streams the FFN matrices and keeps everything else
resident; a stricter budget streams progressively larger sets (for example also the MTP block or
additional layers), and a zero budget streams the entire weight set. This mirrors llama.cpp's
tensor-level offload semantics and gives one familiar knob.

Partial residency (`resident_ffn_layers = N`): the first N text-layer FFN
matrices (indices 0..N-1) are bound device-resident; layers N..63 stream from the
pinned host store through the staging arena. All 64 text-layer FFN matrices are
byte-identical in the registered identities (gate_up 34816x5120 + down 5120x17408;
153,190,400 B groupwise, 150,405,128 B NVFP4 per layer), so "N largest layers"
degenerates to "any N layers", and the deterministic first-N prefix is the exact
behavior. `staged_weights` shrinks to 2*(64-N) entries; the staging arena stays
2 x per-layer-unit regardless of N. Resident layers are skipped by the
`post_mixer` staging gate (`weight.host == nullptr`), so they never receive a
staging copy and their payloads stay in the backing weights arena.

### 4.3 Resident staging arena

A new device `DeviceArena` with fixed addresses holds the streaming slots. Slot capacity is
`2 x largest streaming unit` (double buffering; the double-buffered arena is ~292 MiB groupwise
/ ~287 MiB NVFP4, and the MTP/draft extras stay device-resident under FfnOffload). Every offloaded tensor has one fixed
slot address; resident tensors keep their addresses in the existing backing arena. Kernels and
NVFP4 TMA descriptors are re-pointed to slot addresses at startup; because slot addresses never
change and each round re-sends the same bytes to the same addresses, captured graphs and TMA maps
stay valid across rounds.

### 4.4 Graph restructure

The per-(profile, batch) capture is kept, but the interleave is constructed inside the shared 27B
`post_mixer` leaf rather than in `prepare_graphs()`: the leaf issues a `cudaMemcpyAsync` H2D copy
for each staged gate/up and down before its MLP kernels (gated on `weight.host != nullptr`), so
during capture the copies become in-graph H2D memcpy nodes in front of each offload group's kernel
nodes, and replay re-sends the same bytes to the same fixed slot addresses, so no re-capture. The
double-buffered pipeline (`g % 2` slot alternation established at load) lives inside one captured
graph and the "one graph launch per round" property is preserved.

Prefill is not graph-captured today; the same leaf staging runs eagerly on the round stream before
each group's kernels. Prefill is compute-bound, so streaming is largely hidden.

### 4.5 Capacity planning

`resolve_kv_capacity` and Engine sizing subtract resident weights + staging arena + workspace +
RequestMemory + graph allowance from the card's bytes. KV stays INT8 device-resident and sizes
down via the existing planner; on 16 GB the 27B KV ceiling is reduced from the 32 GB-card maximum.

## 5. Change surfaces

| # | Surface | Files | Change |
|---|---|---|---|
| 1 | Binder/plan | `src/artifact/binder.{h,cpp}`, `src/artifact/materializer.{h,cpp}`, `src/artifact/typed_binding.{h,cpp}` | add `TensorPlacement::Host` for weight tensors, host tensor spans in `MaterializationPlan`, host store in `MaterializedArtifact` + `host_data(handle)`, host-bytes stats |
| 2 | Tensor type | `src/core/tensor.h` | `Tensor`/`Weight` carry an optional host address beside device `data` |
| 3 | 27B bindings | `src/targets/qwen3_6_27b/impl/load/bindings.cpp`, `package.cpp` | residency profile; offloaded tensors bind views to the host store; resident tensors keep the arena |
| 4 | Staging + slot binding | family `model_view.h`, `src/targets/qwen3_6_27b/impl/load/bindings.{h,cpp}` | fixed staging arena; offloaded-tensor -> slot map; NVFP4 TMA maps re-pointed to slots |
| 5 | Graph capture | `src/targets/qwen3_6_27b/impl/variant.cpp`, `src/artifact/materializer.{h,cpp}` | interleave H2D memcpy nodes per offload group inside the shared post_mixer leaf; pinned host store |
| 6 | Engine/sizing | `src/runtime/engine/kv_capacity.cpp`, `include/ninfer/engine.h`, `types.h`, Engine PIMPL | offload option + resident-bytes budget; load/memory summaries report host store and staging |
| 7 | Tests | artifact binder host-placement contract tests, real-artifact load, memory_summary | new host-placement coverage |
| 8 | Docs | `docs/cli.md`, `docs/performance.md`, model cards | option surface and measurement caveat |

Status: surfaces 1-6 are implemented — binder host placement
(`TensorPlacement::Host`, host spans in `MaterializationPlan`), materializer host store
(`MaterializedArtifact::host_data(handle)` + host-bytes stats, pinned via `cudaMallocHost`),
`bind_tensor` host dispatch, `Tensor`/`Weight` host addresses with view propagation, the 27B
`ResidencyProfile` (`AllResident` default, `FfnOffload` binds the per-layer FFN/SwiGLU gate/up +
down matrices host-only), the fixed staging arena with offloaded-tensor -> slot binding (a
`2 x largest-streaming-unit` device `DeviceArena` whose slot addresses never change, the
host-placed `Weight` device planes (`payload`/`qdata`/`qhigh`/`scales`, and by extension the
NVFP4 TMA maps) re-pointed at those slots during `LoadedModelData` construction before graph
capture, and a `ModelView::staged_weights` slot map), the graph-capture staging interleave
(the shared 27B `post_mixer` leaf issues a `cudaMemcpyAsync` host->slot copy for each staged
gate/up and down before its MLP kernels, gated on `weight.host != nullptr`, so the copies become
in-graph H2D memcpy nodes during capture and run eagerly for prefill and non-graph decode), and
the public engine residency option (`EngineOptions::weight_residency`, default `AllResident`;
`FfnOffload` is selectable through the CLI `--weight-residency ffn`), with `LoadSummary` reporting
the pinned host store (`host_bytes`, `host_capacity_bytes`) and `MemorySummary` reporting the
staging arena and `host_store_bytes`. FfnOffload decode runs correctly through the public Engine;
the 35B-A3B target rejects the offload option. A partial-residency knob
(`EngineOptions::resident_ffn_layers`, default 0 = all
stream; CLI `--n-ffn-layers N`) keeps the first N FFN text layers device-resident
and validates `N <= 64` in the 27B binder; MTP3 under FfnOffload is verified
end-to-end.

Unchanged: KV cache, workspace, RequestMemory, scheduler round logic, kernels, media/frontend
paths.

## 6. Expected performance

Every decode round re-reads every non-resident weight (inherent to a dense model). On a PCIe 5.0
x8 16 GB card (~30 GB/s effective H2D):

- Fully streamed (~17.07 GiB/round): ~1.8-2 tok/s (MTP0).
- FFN-offload profile (~10.2 GiB/round streamed, attention/embeddings/head resident): ~2.9-3.4
  tok/s.
- FFN-offload plus some resident FFN layers: ~4-5 tok/s.

Reference: the 32 GiB 5090 resident decode is 54-77 tok/s (MTP0); offload is 20-30x slower,
consistent with llama.cpp's real-world 27B-offloaded numbers. Prefill is largely unaffected
(compute-bound; streaming overlaps).

## 7. 35B-A3B (bonus, not primary scope)

The streaming framework (sections 4.1, 4.3-4.5, change surfaces 1, 2, 4-6) is shared, but 35B-A3B
adds real work beyond 27B: MoE needs the 8 selected experts per layer before copying them (~14 MB/
layer streamed instead of the full ~18.4 GiB expert bank), which requires a per-layer device-to-host
router read-back inside or between graphs. The current single-graph-per-profile capture cannot do
that without per-layer segmentation or conditional nodes. Treat as separate scope.

## 8. Risks and open items

- Host RAM: ~17 GiB pinned requires a 32+ GiB host; pinning is required for overlapping DMA
  (pageable H2D falls back to staged copies).
- In-graph H2D memcpy nodes are supported and deterministic; still needs a capture-time validation
  test.
- 16 GB KV ceiling: at the full 262144 context the INT8 KV can reach ~8.6 GiB; offload mode sizes
  context down through the existing planner.
- NVFP4: TMA maps must be re-encoded against slot addresses at startup, not per round.
- Vision and DFlash weights are separate stream units exercised only on request begin.
