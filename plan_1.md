  ## Plan: MXFP4 Support Discovery and SM120-First Bring-Up (GPT-OSS, max.nn path)

  ### Summary

  - Goal: determine exactly what is required to run MXFP4 checkpoints (for example GPT-OSS 20B) on your SM120 GPU, then implement the smallest working path starting with a standalone
    MXFP4 matmul kernel.
  - Strategy: treat MXFP4 as a checkpoint/data format + scaling scheme first, not as a required single “MXFP4 instruction.” Use a correctness-first fallback kernel on SM120, then
    optimize later.
  - First deliverable: standalone MXFP4 matmul kernel + tests, then wire into gpt_oss (not gpt_oss_modulev3).

  ### Phase 1: Fact-Finding and Spec Lock (No model integration yet)

  1. Create an internal design note (docs/eng-design/mxfp4-sm120-plan.md) that locks:

  - MXFP4 weight packing layout.
  - Scale dtype/layout and scale application rule.
  - Per-layer tensor shapes for GPT-OSS MoE and dense linears.
  - Required numerical behavior (reference dequant math).

  2. Validate hardware/ISA constraints explicitly:

  - Record that existing MAX NVFP4 fast kernels are SM100-specific (tcgen05/block-scaled path in max/kernels/src/linalg/grouped_matmul_sm100_1d1d.mojo and max/kernels/src/linalg/
    fp4_quantization.mojo).
  - Record that GPT-OSS architectures are currently BF16-only in:
      - max/python/max/pipelines/architectures/gpt_oss/arch.py
      - max/python/max/pipelines/architectures/gpt_oss_modulev3/arch.py

  3. Add a lightweight inspection script (non-runtime dependency) to parse target checkpoint metadata and safetensor headers:

  - Confirm whether checkpoint uses quant_algo: MXFP4, scale tensor names, and exact dtypes.
  - Emit a JSON report consumed by tests.

  ### Phase 2: Standalone MXFP4 Kernel Milestone (SM120 correctness-first)

  1. Add new Mojo custom ops under MAX kernels (new files in max/kernels/src/linalg/):

  - mo.mxfp4.unpack (packed FP4 + scales -> BF16/FP32 tensor).
  - mo.mxfp4.matmul.reference (MXFP4 weights + BF16 activations -> BF16 output).

  2. Implement as fallback arithmetic kernels (no tcgen05 requirement):

  - Unpack nibble pairs from uint8.
  - Convert FP4 E2M1 values via LUT.
  - Apply per-block/per-group scale as defined by checkpoint spec.
  - Multiply-accumulate in FP32, cast output to BF16.

  3. Register Python wrappers in max/python/max/nn/kernels.py:

  - mxfp4_unpack(...)
  - mxfp4_matmul_reference(...)
  - Keep signatures explicit about expected tensor ranks/layouts.

  4. Gate behavior by capability:

  - On SM100 path: keep existing NVFP4 fast kernels untouched.
  - On SM120 path: use new reference/fallback path.
  - If unsupported shape/layout, raise clear ValueError with required constraints.

  ### Phase 3: Integrate into GPT-OSS (max.nn path only)

  1. Enable encoding in GPT-OSS architecture:

  - Extend supported_encodings to include "float4_e2m1fnx2" in max/python/max/pipelines/architectures/gpt_oss/arch.py.

  2. Extend config parsing path for GPT-OSS:

  - Parse float4 config in model setup (same pattern used in DeepSeek/Llama codepaths).
  - Thread float8_config through GPT-OSS config/model objects where needed.

  3. Update GPT-OSS layers to use quantized path:

  - Dense projections: use max.nn.linear.Linear(..., float8_config=...) where applicable.
  - MoE path: use MoEQuantized/NVFP4 strategy-compatible route where tensor/layout contracts match.
  - For any mismatch with GPT-OSS checkpoint layout, add explicit adapter transforms in weight adapter layer.

  4. Keep gpt_oss_modulev3 out of initial scope.

  - Document gap and create follow-up task after gpt_oss path is stable.

  ### Public API / Interface Changes

  1. Add new custom op names:

  - mo.mxfp4.unpack
  - mo.mxfp4.matmul.reference

  2. Add Python kernel wrappers in max.nn.kernels for these ops.
  3. GPT-OSS architecture support update:

  - supported_encodings includes "float4_e2m1fnx2" for GptOssForCausalLM.

  4. Add internal checkpoint-inspection utility output format (JSON schema documented in the design note).

  ### Test Plan and Acceptance Criteria

  1. Kernel correctness tests:

  - Synthetic tensor tests for unpack and matmul against CPU reference implementation.
  - Shape edge cases: non-multiple K groups, ragged group counts, empty batch.

  2. Capability tests:

  - Ensure SM120 selects fallback kernels.
  - Ensure SM100 path remains unchanged.

  3. Integration tests:

  - GPT-OSS config parsing accepts MXFP4 metadata and builds model graph.
  - One end-to-end smoke test (short prompt, few tokens) on SM120 with MXFP4 checkpoint.

  4. Regression tests:

  - Existing NVFP4 tests remain green (max/tests/tests/kv_cache/test_fp4_matmul.py, related integration tests).

  ### Assumptions and Defaults

  - In-scope hardware: SM120 first (your available GPU).
  - First milestone optimization target: correctness over throughput.
  - Initial integration target: gpt_oss (max.nn) path only.
  - No third-party runtime backend dependency in first milestone.
  - Existing SM100 NVFP4 fast paths are preserved and not refactored in this effort.

  ### External References Used

  - NVIDIA PTX ISA (tcgen05 and block scaling context): https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
  - NVIDIA Blackwell architecture guide: https://docs.nvidia.com/cuda/blackwell-compatibility-guide/index.html
  - NVIDIA Transformer Engine FP4 primer: https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.6/user-guide/examples/fp8_primer.html
  - NVIDIA NIM GPT-OSS model card (MXFP4 checkpoint statement): https://build.nvidia.com/openai/gpt-oss-20b
  - Reference implementation you shared: https://github.com/RWayne93/gpt-oss/blob/mojo-backend/gpt_oss/mojo/operations/mxfp4.mojo
  - vLLM MXFP4 backend selection and utils:
      - https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/mxfp4.py
      - https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/utils/mxfp4_utils.py

  ### Addendum: Kernel Placement and Fusion Pattern (from repository scan)

  1. Where fusion is implemented today (prevailing pattern):

  - Python graph wrappers live in `max/python/max/nn/kernels.py` and call `ops.custom(...)`.
    - Current grouped NVFP4 entrypoint: `grouped_dynamic_scaled_nvfp4_matmul` (`max/python/max/nn/kernels.py:3154`).
  - Custom-op registrations for graph compiler live in `max/kernels/src/Mogg/MOGGKernelAPI/`.
    - Grouped NVFP4 registration: `mo.grouped.matmul.dynamic.scaled.nvfp4` in `MOGGKernelAPI.mojo:8096`.
    - EP-specific fused ops are added in dedicated files (for example `ep_api.mojo`), not only in the monolithic file.
  - Actual high-performance kernels live under `max/kernels/src/linalg/...` (SM100 structured stack).
    - Grouped NVFP4 kernel wrapper: `grouped_matmul_dynamic_scaled_nvfp4` in `grouped_1d1d_matmul.mojo:418`.
    - Core grouped block-scaled kernel type: `grouped_1d1d_matmul_kernel.mojo`.
  - EP MoE fusion (dispatch/combine + fused SiLU quantization) lives in `shmem/ep_comm.mojo` and is exposed via `MOGGKernelAPI/ep_api.mojo`.
    - `NVFP4TokenFormat`: `shmem/ep_comm.mojo:667`
    - `fused_silu_nvfp4_kernel`: `shmem/ep_comm.mojo:3308`
    - API registration: `ep.fused_silu.nvfp4` in `ep_api.mojo:1397`
  - MLA SM100 follows a dispatcher pattern (thin frontdoor + specialized impl files), useful as style reference but not the right insertion point for MoE grouped matmul:
    - `mla_sm100_prefill` dispatcher: `nn/mla_prefill_sm100.mojo:30`
    - `mla_decode_sm100_dispatch`: `nn/mla_decode_sm100_dispatch.mojo:59`

  2. Recommended insertion point for fast MXFP4 MoE kernel:

  - Primary implementation should be in:
    - `max/kernels/src/linalg/matmul/gpu/sm100_structured/grouped_block_scaled_1d1d/grouped_1d1d_matmul.mojo`
    - `max/kernels/src/linalg/matmul/gpu/sm100_structured/structured_kernels/config.mojo`
    - potentially `max/kernels/src/linalg/arch/sm100/mma.mojo` (scale-tile indexing rules for FP4 + vec32 scales)
  - Then expose through a dedicated MOGG API file in:
    - `max/kernels/src/Mogg/MOGGKernelAPI/` (same style as `ep_api.mojo`/`mxfp4_api.mojo`)
  - Then wire in Python:
    - `max/python/max/nn/kernels.py`
    - `max/python/max/nn/moe/quant_strategy.py`
    - `max/python/max/nn/moe/moe_fp8.py`
    - `max/python/max/nn/float8_config.py`

  3. Concrete implementation steps (fast path, aligned with existing style):

  - Generalize grouped FP4 kernel entrypoint:
    - Add `grouped_matmul_dynamic_scaled_fp4` (or equivalent) beside existing NVFP4 entrypoint in `grouped_1d1d_matmul.mojo`.
    - Accept FP4 packed inputs (`uint8`) with either NVFP4 scales (`float8_e4m3fn`, vec16) or MXFP4 scales (`float8_e8m0fnu`, vec32).
  - Update structured config to derive scale-vector behavior from scale dtype (not only scaling kind):
    - current hard-coding is in `structured_kernels/config.mojo` (`vec_sf_size`/`num_sf_k_tiles` around lines 616-624).
  - Update SM100 MMA path if needed for vec32 FP4 scales:
    - current FP4 path assumes one scale tile per `k` iteration (comment around `mma.mojo:851`).
    - MXFP4 likely needs different scale-tile reuse/indexing than NVFP4.
  - Add new custom op registration in `MOGGKernelAPI`:
    - new op name should be explicit (for example `mo.grouped.matmul.dynamic.scaled.mxfp4`) or a unified fp4 op with runtime dtype checks.
    - Avoid editing the monolithic generated file unless required; prefer a dedicated API file.
  - Add Python wrappers and strategy routing:
    - keep existing NVFP4 path unchanged.
    - add explicit MXFP4 path and validation.
  - Add tests:
    - Unit shape/dtype checks in `max/tests/tests/kv_cache/test_fp4_matmul.py`.
    - Integration correctness/perf in `max/tests/integration/nn/` (similar style to `test_mxfp4_reference_gpu.py` and EP FP4 tests).

  4. Benchmarking path for this fast kernel work:

  - Use `max/kernels/benchmarks/gpu/bench_grouped_matmul.mojo` (has explicit NVFP4 grouped path at lines 283-452).
  - Add/duplicate benchmark mode for MXFP4 to compare:
    - reference path vs fused kernel
    - NVFP4 fused vs MXFP4 fused for matched shapes
    - MoE-like shapes (token imbalance per expert)

  ### Addendum: SM120 Reality Check (important)

  1. `sm100_structured` is not just a name in current tree.

  - It is wired to SM100/B200-specific assumptions in multiple places:
    - imports and types from `mma_nvidia_sm100` + tcgen05 paths
    - explicit `B200` constants used for launch and shared-memory sizing
    - examples:
      - `grouped_1d1d_matmul.mojo` uses `B200.sm_count` for `grid_dim`
      - `structured_kernels/config.mojo` imports/uses `B200` properties
      - `fp4_quantization.mojo` has `ctx.default_device_info.compute == B200.compute` asserts

  2. Implication for SM120 implementation:

  - Do not rely on current `sm100_structured` grouped FP4 path as the first SM120 fast path.
  - Treat existing `sm100_structured` kernels as B200-tuned and potentially non-portable until explicitly generalized.

  3. Preferred SM120-first fast path:

  - Keep current correctness oracle (`mxfp4_unpack` + reference matmul) as baseline.
  - Implement first optimized path using vendor block-scaled matmul (`cublasLt`) where possible:
    - generalize the B200 gate in `linalg/matmul/vendor/blas.mojo` to accept newer Blackwell compute targets
    - add support for MXFP4 contracts (packed FP4 input + vec32 UE8M0 scales) in that vendor path
  - For grouped MoE execution:
    - add a temporary SM120 fallback that is correctness-preserving (even if slower), then iterate to fused grouped kernel.

  4. Longer-term kernelization for SM120:

  - After vendor-backed path is validated, either:
    - generalize `sm100_structured` code to “blackwell_structured” (remove B200 hard-coding, runtime device params), or
    - add dedicated `sm120` kernel path and dispatch.

  ### Addendum: Next Execution Plan (March 2026 update)

  #### Phase 5: GPT-OSS MXFP4 Bring-Up (Runability First)

  1. Architecture/config acceptance:
  - Add `"float4_e2m1fnx2"` to GPT-OSS supported encodings.
  - Parse `Float8Config`/modelopt FP4 config for GPT-OSS during model build (same style as Llama3/DeepSeek paths).
  - Thread `float8_config` through GPT-OSS config and layer construction.

  2. Dense-path enablement:
  - Route GPT-OSS attention QKV/O projections through FP4 path when `float8_config.is_mxfp4`.
  - Keep SM120 on correctness fallback (`naive_block_scaled_matmul`) via existing dispatch gate.

  3. Explicit MoE limitation handling:
  - Add a clear, early error for stacked GPT-OSS MoE + MXFP4 so failures are actionable.
  - Do not silently fall back to incorrect FP8 assumptions.

  4. Validation:
  - Add/extend GPT-OSS config tests to verify FP4 config parsing and graph build path selection.
  - Run targeted tests for FP4 kernels + GPT-OSS config build.

  #### Phase 6: SM120 MXFP4 MoE Fallback Kernel (Correctness)

  1. Implement a non-tcgen05 grouped MXFP4 MoE path:
  - New grouped FP4 reference op/path that supports ragged expert routing with packed uint8 + vec32 e8m0 scales.
  - Use it only when tcgen05 grouped kernels are unavailable (SM120 path today).

  2. Wire MoE strategy:
  - Add MXFP4 quant strategy path in `max.nn.moe` and GPT-OSS stacked MoE callsite routing.
  - Preserve existing NVFP4 fast path behavior.

  3. Correctness tests:
  - Compare grouped MXFP4 fallback outputs vs dequantized BF16 reference for MoE-like shapes.

  ### Status Update (March 3, 2026)

  Completed in this branch:
  - Grouped MXFP4 fallback path in `max.nn.kernels` and MoE quant strategy wiring (`Mxfp4Strategy`).
  - `MoEQuantized` path now supports MXFP4 in non-EP mode (EP fused MXFP4 remains intentionally unimplemented).
  - GPT-OSS float4 config parsing now accepts HuggingFace `quant_method: "mxfp4"` and normalizes to modelopt metadata.
  - GPT-OSS layer construction now respects `mlp_in_float8` / `attn_qkv_in_float8` sets (so unconverted attention stays BF16).
  - GPT-OSS weight adapter now handles OpenAI checkpoint MoE tensors:
    - `*_blocks` -> flattened packed FP4 tensors
    - `*_scales` (UE8M0 bytes) -> reinterpreted as `float8_e8m0fnu`
  - StackedMoE now has an MXFP4/NVFP4 quantized forward branch (correctness fallback) using grouped FP4 kernels.
  - Added/ran GPU smoke coverage for stacked MXFP4 MoE fallback (`test_stacked_moe_mxfp4_basic`).

  Remaining for "fast path" objective:
  - Implement a true optimized grouped MXFP4 kernel path for SM120 (not the dequantize+BF16 fallback).
  - Remove/avoid temporary FP4 TP limitation in `StackedMoE` (`num_devices > 1` currently blocked for FP4).
  - Add benchmark-driven validation against GPT-OSS throughput targets and identify hottest bottlenecks.

  #### Phase 7: Performance Work (After Correct End-to-End)

  1. Baseline measurements:
  - Benchmark token/s and per-layer latency on SM120 using current fallback.
  - Benchmark isolated grouped MoE kernels with MoE-representative shapes.

  2. Fast path implementation:
  - Add SM120 optimized grouped FP4 kernel path (vendor-backed or dedicated kernel path).
  - Fuse SiLU+quantize+down-proj where practical, following existing EP/NVFP4 fusion structure.

  3. Exit criteria:
  - GPT-OSS 20B MXFP4 end-to-end generation on SM120.
  - Correctness parity against reference path.
  - Measurable throughput improvement over fallback baseline.
