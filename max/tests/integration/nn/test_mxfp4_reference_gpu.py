# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import os
import re
import subprocess
import time
from typing import Any

import pytest
import torch
from max.driver import Buffer, accelerator_api, accelerator_architecture_name
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn.kernels import (
    dynamic_block_scaled_matmul_fp4,
    dynamic_block_scaled_matmul_fp4_reference,
    grouped_dynamic_scaled_mxfp4_matmul,
    mxfp4_unpack,
)

ABS_TOL_UNPACK = 5e-2
ABS_TOL_MATMUL = 2e-1
RTOL = 1e-2


def _is_sm100_or_newer() -> bool:
    if accelerator_api() == "hip":
        return False
    arch = accelerator_architecture_name()
    match = re.search(r"sm_?(\d+)", arch)
    if match is not None:
        return int(match.group(1)) >= 100

    # Fallback: parse compute capability from nvidia-smi
    # (e.g. "12.0" for SM120, "10.0" for SM100).
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=compute_cap",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        first_line = result.stdout.strip().splitlines()[0]
        major = int(first_line.split(".", 1)[0])
        return major >= 10
    except (IndexError, ValueError, subprocess.SubprocessError, OSError):
        return False


def _ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _scale_shape(m: int, k: int, sf_vector_size: int) -> tuple[int, int, int, int, int]:
    return (
        _ceildiv(m, 128),
        _ceildiv(k, sf_vector_size * 4),
        32,
        4,
        4,
    )


def _fp4_lut(device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        dtype=torch.float32,
        device=device,
    )


def _unpack_reference(
    packed: torch.Tensor,
    scales_fp32: torch.Tensor,
    sf_vector_size: int,
) -> torch.Tensor:
    m, packed_k = packed.shape
    k = packed_k * 2
    device = packed.device

    byte = packed.to(torch.uint8)
    lo = byte & 0x0F
    hi = byte >> 4
    nibbles = torch.stack((lo, hi), dim=-1).reshape(m, k).to(torch.int64)
    values = _fp4_lut(device)[nibbles]

    row = torch.arange(m, device=device).view(m, 1)
    col = torch.arange(k, device=device).view(1, k)
    scale_vals = scales_fp32[
        row // 128,
        col // (sf_vector_size * 4),
        row % 32,
        (row % 128) // 32,
        (col // sf_vector_size) % 4,
    ]

    return (values * scale_vals.abs()).to(torch.bfloat16)


def _to_max_fp8_e4m3fn(x: torch.Tensor) -> Buffer:
    # DLPack float8 interop is not available; pass raw bytes then reinterpret.
    return Buffer.from_dlpack(x.view(torch.uint8)).view(
        DType.float8_e4m3fn, x.shape
    )


def _to_max_fp8_e8m0fnu(x: torch.Tensor) -> Buffer:
    # DLPack float8 interop is not available; pass raw bytes then reinterpret.
    return Buffer.from_dlpack(x.view(torch.uint8)).view(
        DType.float8_e8m0fnu, x.shape
    )


def _to_max_scale_buffer(x: torch.Tensor, scales_dtype: DType) -> Buffer:
    if scales_dtype == DType.float8_e4m3fn:
        return _to_max_fp8_e4m3fn(x)
    if scales_dtype == DType.float8_e8m0fnu:
        return _to_max_fp8_e8m0fnu(x)
    raise ValueError(f"unsupported scales dtype: {scales_dtype}")


def _build_unpack_model(
    session: InferenceSession,
    m: int,
    packed_k: int,
    sf_shape: tuple[int, int, int, int, int],
    sf_vector_size: int,
    scales_dtype: DType,
) -> Any:
    with Graph(
        "mxfp4_unpack_ref_gpu",
        input_types=[
            TensorType(DType.uint8, [m, packed_k], device=DeviceRef.GPU()),
            TensorType(scales_dtype, list(sf_shape), device=DeviceRef.GPU()),
        ],
    ) as graph:
        packed, scales = (v.tensor for v in graph.inputs)
        graph.output(mxfp4_unpack(packed, scales, sf_vector_size=sf_vector_size))

    return session.load(graph)


def _build_reference_matmul_model(
    session: InferenceSession,
    m: int,
    n: int,
    packed_k: int,
    a_sf_shape: tuple[int, int, int, int, int],
    b_sf_shape: tuple[int, int, int, int, int],
    sf_vector_size: int,
    scales_dtype: DType,
) -> Any:
    with Graph(
        "mxfp4_matmul_ref_gpu",
        input_types=[
            TensorType(DType.uint8, [m, packed_k], device=DeviceRef.GPU()),
            TensorType(DType.uint8, [n, packed_k], device=DeviceRef.GPU()),
            TensorType(scales_dtype, list(a_sf_shape), device=DeviceRef.GPU()),
            TensorType(scales_dtype, list(b_sf_shape), device=DeviceRef.GPU()),
            TensorType(DType.float32, [], device=DeviceRef.GPU()),
        ],
    ) as graph:
        a, b, a_sf, b_sf, tensor_sf = (v.tensor for v in graph.inputs)
        graph.output(
            dynamic_block_scaled_matmul_fp4_reference(
                a,
                b,
                a_sf,
                b_sf,
                tensor_sf,
                sf_vector_size=sf_vector_size,
                out_type=DType.bfloat16,
            )
        )

    return session.load(graph)


def _build_optimized_matmul_model(
    session: InferenceSession,
    m: int,
    n: int,
    packed_k: int,
    a_sf_shape: tuple[int, int, int, int, int],
    b_sf_shape: tuple[int, int, int, int, int],
    sf_vector_size: int,
    scales_dtype: DType,
) -> Any:
    with Graph(
        "mxfp4_matmul_optimized_gpu",
        input_types=[
            TensorType(DType.uint8, [m, packed_k], device=DeviceRef.GPU()),
            TensorType(DType.uint8, [n, packed_k], device=DeviceRef.GPU()),
            TensorType(scales_dtype, list(a_sf_shape), device=DeviceRef.GPU()),
            TensorType(scales_dtype, list(b_sf_shape), device=DeviceRef.GPU()),
            TensorType(DType.float32, [], device=DeviceRef.GPU()),
        ],
    ) as graph:
        a, b, a_sf, b_sf, tensor_sf = (v.tensor for v in graph.inputs)
        graph.output(
            dynamic_block_scaled_matmul_fp4(
                a,
                b,
                a_sf,
                b_sf,
                tensor_sf,
                sf_vector_size=sf_vector_size,
                out_type=DType.bfloat16,
            )
        )

    return session.load(graph)


def _build_baseline_matmul_model(
    session: InferenceSession,
    m: int,
    n: int,
    k: int,
) -> Any:
    with Graph(
        "mxfp4_matmul_baseline_gpu",
        input_types=[
            TensorType(DType.bfloat16, [m, k], device=DeviceRef.GPU()),
            TensorType(DType.bfloat16, [n, k], device=DeviceRef.GPU()),
            TensorType(DType.float32, [], device=DeviceRef.GPU()),
        ],
    ) as graph:
        a, b, tensor_sf = (v.tensor for v in graph.inputs)
        result = ops.matmul(a, ops.transpose(b, 0, 1))
        graph.output(result * tensor_sf)

    return session.load(graph)


def _build_grouped_mxfp4_model(
    session: InferenceSession,
    total_tokens: int,
    num_experts: int,
    n: int,
    packed_k: int,
    a_sf_shape: tuple[int, int, int, int, int],
    b_sf_shape: tuple[int, int, int, int, int, int],
) -> Any:
    with Graph(
        "mxfp4_grouped_matmul_gpu",
        input_types=[
            TensorType(
                DType.uint8, [total_tokens, packed_k], device=DeviceRef.GPU()
            ),
            TensorType(
                DType.uint8,
                [num_experts, n, packed_k],
                device=DeviceRef.GPU(),
            ),
            TensorType(
                DType.float8_e8m0fnu, list(a_sf_shape), device=DeviceRef.GPU()
            ),
            TensorType(
                DType.float8_e8m0fnu, list(b_sf_shape), device=DeviceRef.GPU()
            ),
            TensorType(
                DType.uint32, [num_experts + 1], device=DeviceRef.GPU()
            ),
            TensorType(DType.uint32, [num_experts], device=DeviceRef.GPU()),
            TensorType(DType.int32, [num_experts], device=DeviceRef.GPU()),
            TensorType(DType.float32, [num_experts], device=DeviceRef.GPU()),
            TensorType(DType.uint32, [2], device=DeviceRef.GPU()),
        ],
    ) as graph:
        (
            hidden,
            weights,
            a_sf,
            b_sf,
            expert_start,
            a_scale_offsets,
            expert_ids,
            expert_scales,
            usage_stats,
        ) = (v.tensor for v in graph.inputs)
        graph.output(
            grouped_dynamic_scaled_mxfp4_matmul(
                hidden,
                weights,
                a_sf,
                b_sf,
                expert_start,
                a_scale_offsets,
                expert_ids,
                expert_scales,
                usage_stats,
            )
        )

    return session.load(graph)


def _run_timed(
    model: Any,
    args: tuple[Buffer, ...],
    device: Any,
    warmup: int,
    iters: int,
) -> float:
    for _ in range(warmup):
        model.execute(*args)
    device.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        model.execute(*args)
    device.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / iters


@pytest.mark.skipif(
    accelerator_api() == "hip", reason="Test is NVIDIA-specific for now"
)
@pytest.mark.parametrize(
    ("sf_vector_size", "scales_dtype", "torch_scale_dtype"),
    [
        (16, DType.float8_e4m3fn, torch.float8_e4m3fn),
        (
            32,
            DType.float8_e8m0fnu,
            getattr(torch, "float8_e8m0fnu", None),
        ),
    ],
)
def test_mxfp4_unpack_matches_reference(
    gpu_session: InferenceSession,
    sf_vector_size: int,
    scales_dtype: DType,
    torch_scale_dtype: torch.dtype | None,
) -> None:
    if torch_scale_dtype is None:
        pytest.skip("torch.float8_e8m0fnu is unavailable in this environment")

    torch.manual_seed(7)
    m = 96
    k = 256
    packed_k = k // 2
    sf_shape = _scale_shape(m, k, sf_vector_size)
    device = torch.device("cuda")

    packed = torch.randint(
        0, 256, (m, packed_k), dtype=torch.uint8, device=device
    )
    scales_fp32 = torch.rand(sf_shape, dtype=torch.float32, device=device) + 0.1
    scales_fp8 = scales_fp32.to(torch_scale_dtype)

    model = _build_unpack_model(
        gpu_session, m, packed_k, sf_shape, sf_vector_size, scales_dtype
    )
    result = torch.from_dlpack(
        model.execute(
            Buffer.from_dlpack(packed),
            _to_max_scale_buffer(scales_fp8, scales_dtype),
        )[0]
    ).to(torch.float32)
    expected = _unpack_reference(
        packed, scales_fp8.to(torch.float32), sf_vector_size
    ).to(torch.float32)

    torch.testing.assert_close(
        result, expected, rtol=RTOL, atol=ABS_TOL_UNPACK
    )


@pytest.mark.skipif(
    accelerator_api() == "hip", reason="Test is NVIDIA-specific for now"
)
@pytest.mark.parametrize(
    ("sf_vector_size", "scales_dtype", "torch_scale_dtype"),
    [
        (16, DType.float8_e4m3fn, torch.float8_e4m3fn),
        (
            32,
            DType.float8_e8m0fnu,
            getattr(torch, "float8_e8m0fnu", None),
        ),
    ],
)
def test_mxfp4_reference_matmul_matches_reference(
    gpu_session: InferenceSession,
    sf_vector_size: int,
    scales_dtype: DType,
    torch_scale_dtype: torch.dtype | None,
) -> None:
    if torch_scale_dtype is None:
        pytest.skip("torch.float8_e8m0fnu is unavailable in this environment")

    torch.manual_seed(11)
    m = 128
    n = 160
    k = 256
    packed_k = k // 2
    a_sf_shape = _scale_shape(m, k, sf_vector_size)
    b_sf_shape = _scale_shape(n, k, sf_vector_size)
    device = torch.device("cuda")

    a = torch.randint(0, 256, (m, packed_k), dtype=torch.uint8, device=device)
    b = torch.randint(0, 256, (n, packed_k), dtype=torch.uint8, device=device)
    a_scales_fp32 = torch.rand(a_sf_shape, dtype=torch.float32, device=device) + 0.1
    b_scales_fp32 = torch.rand(b_sf_shape, dtype=torch.float32, device=device) + 0.1
    a_scales_fp8 = a_scales_fp32.to(torch_scale_dtype)
    b_scales_fp8 = b_scales_fp32.to(torch_scale_dtype)
    tensor_sf = torch.tensor(0.125, dtype=torch.float32, device=device)

    model = _build_reference_matmul_model(
        gpu_session,
        m,
        n,
        packed_k,
        a_sf_shape,
        b_sf_shape,
        sf_vector_size,
        scales_dtype,
    )
    result = torch.from_dlpack(
        model.execute(
            Buffer.from_dlpack(a),
            Buffer.from_dlpack(b),
            _to_max_scale_buffer(a_scales_fp8, scales_dtype),
            _to_max_scale_buffer(b_scales_fp8, scales_dtype),
            Buffer.from_dlpack(tensor_sf),
        )[0]
    ).to(torch.float32)

    a_deq = _unpack_reference(a, a_scales_fp8.to(torch.float32), sf_vector_size)
    b_deq = _unpack_reference(b, b_scales_fp8.to(torch.float32), sf_vector_size)
    expected = (a_deq.to(torch.float32) @ b_deq.to(torch.float32).T) * tensor_sf

    torch.testing.assert_close(
        result, expected, rtol=RTOL, atol=ABS_TOL_MATMUL
    )


def test_grouped_mxfp4_fallback_matches_reference(
    gpu_session: InferenceSession,
) -> None:
    if not hasattr(torch, "float8_e8m0fnu"):
        pytest.skip("torch.float8_e8m0fnu is unavailable in this environment")

    torch.manual_seed(97)
    total_tokens = 48
    num_experts = 3
    n = 96
    k = 256
    sf_vector_size = 32
    packed_k = k // 2
    device = torch.device("cuda")

    hidden = torch.randint(
        0, 256, (total_tokens, packed_k), dtype=torch.uint8, device=device
    )
    weights = torch.randint(
        0,
        256,
        (num_experts, n, packed_k),
        dtype=torch.uint8,
        device=device,
    )
    a_sf_shape = _scale_shape(total_tokens, k, sf_vector_size)
    b_sf_shape = (num_experts, *_scale_shape(n, k, sf_vector_size))
    a_scales_fp8 = (
        torch.rand(a_sf_shape, dtype=torch.float32, device=device) + 0.1
    ).to(torch.float8_e8m0fnu)
    b_scales_fp8 = (
        torch.rand(b_sf_shape, dtype=torch.float32, device=device) + 0.1
    ).to(torch.float8_e8m0fnu)

    # Three expert groups over the token-major packed inputs.
    expert_start = torch.tensor([0, 18, 32, 48], dtype=torch.uint32, device=device)
    expert_ids = torch.tensor([2, 0, 1], dtype=torch.int32, device=device)
    a_scale_offsets = torch.zeros(num_experts, dtype=torch.uint32, device=device)
    expert_scales = torch.rand(num_experts, dtype=torch.float32, device=device) + 0.2
    usage_stats = torch.tensor([18, 3], dtype=torch.uint32, device=device)

    model = _build_grouped_mxfp4_model(
        gpu_session,
        total_tokens,
        num_experts,
        n,
        packed_k,
        a_sf_shape,
        b_sf_shape,
    )
    result = torch.from_dlpack(
        model.execute(
            Buffer.from_dlpack(hidden),
            Buffer.from_dlpack(weights),
            _to_max_scale_buffer(a_scales_fp8, DType.float8_e8m0fnu),
            _to_max_scale_buffer(b_scales_fp8, DType.float8_e8m0fnu),
            Buffer.from_dlpack(expert_start),
            Buffer.from_dlpack(a_scale_offsets),
            Buffer.from_dlpack(expert_ids),
            Buffer.from_dlpack(expert_scales),
            Buffer.from_dlpack(usage_stats),
        )[0]
    ).to(torch.float32)

    hidden_deq = _unpack_reference(
        hidden, a_scales_fp8.to(torch.float32), sf_vector_size
    ).to(torch.float32)
    weight_deq = [
        _unpack_reference(
            weights[e], b_scales_fp8[e].to(torch.float32), sf_vector_size
        ).to(torch.float32)
        * expert_scales[e].to(torch.float32)
        for e in range(num_experts)
    ]

    expected = torch.zeros((total_tokens, n), dtype=torch.float32, device=device)
    for group_idx in range(num_experts):
        start = int(expert_start[group_idx].item())
        end = int(expert_start[group_idx + 1].item())
        expert_id = int(expert_ids[group_idx].item())
        expected[start:end] = (
            hidden_deq[start:end] @ weight_deq[expert_id].transpose(0, 1)
        )

    torch.testing.assert_close(result, expected, rtol=RTOL, atol=ABS_TOL_MATMUL)


@pytest.mark.skipif(
    accelerator_api() == "hip", reason="Test is NVIDIA-specific for now"
)
@pytest.mark.skipif(
    not _is_sm100_or_newer(),
    reason="Optimized block-scaled FP4 matmul requires SM100+",
)
@pytest.mark.parametrize(
    ("sf_vector_size", "scales_dtype", "torch_scale_dtype"),
    [
        (16, DType.float8_e4m3fn, torch.float8_e4m3fn),
        (
            32,
            DType.float8_e8m0fnu,
            getattr(torch, "float8_e8m0fnu", None),
        ),
    ],
)
def test_mxfp4_optimized_matmul_matches_reference(
    gpu_session: InferenceSession,
    sf_vector_size: int,
    scales_dtype: DType,
    torch_scale_dtype: torch.dtype | None,
) -> None:
    if torch_scale_dtype is None:
        pytest.skip("torch.float8_e8m0fnu is unavailable in this environment")

    torch.manual_seed(17)
    m = 128
    n = 128
    k = 256
    packed_k = k // 2
    a_sf_shape = _scale_shape(m, k, sf_vector_size)
    b_sf_shape = _scale_shape(n, k, sf_vector_size)
    device = torch.device("cuda")

    a = torch.randint(0, 256, (m, packed_k), dtype=torch.uint8, device=device)
    b = torch.randint(0, 256, (n, packed_k), dtype=torch.uint8, device=device)
    a_scales_fp32 = torch.rand(a_sf_shape, dtype=torch.float32, device=device) + 0.1
    b_scales_fp32 = torch.rand(b_sf_shape, dtype=torch.float32, device=device) + 0.1
    a_scales_fp8 = a_scales_fp32.to(torch_scale_dtype)
    b_scales_fp8 = b_scales_fp32.to(torch_scale_dtype)
    tensor_sf = torch.tensor(0.25, dtype=torch.float32, device=device)

    optimized = _build_optimized_matmul_model(
        gpu_session,
        m,
        n,
        packed_k,
        a_sf_shape,
        b_sf_shape,
        sf_vector_size,
        scales_dtype,
    )
    result = torch.from_dlpack(
        optimized.execute(
            Buffer.from_dlpack(a),
            Buffer.from_dlpack(b),
            _to_max_scale_buffer(a_scales_fp8, scales_dtype),
            _to_max_scale_buffer(b_scales_fp8, scales_dtype),
            Buffer.from_dlpack(tensor_sf),
        )[0]
    ).to(torch.float32)

    a_deq = _unpack_reference(a, a_scales_fp8.to(torch.float32), sf_vector_size)
    b_deq = _unpack_reference(b, b_scales_fp8.to(torch.float32), sf_vector_size)
    expected = (a_deq.to(torch.float32) @ b_deq.to(torch.float32).T) * tensor_sf

    torch.testing.assert_close(
        result, expected, rtol=RTOL, atol=ABS_TOL_MATMUL
    )


@pytest.mark.skipif(
    accelerator_api() == "hip", reason="Test is NVIDIA-specific for now"
)
@pytest.mark.skipif(
    os.getenv("RUN_PERF_TESTS") != "1",
    reason="Set RUN_PERF_TESTS=1 to enable perf check",
)
def test_mxfp4_reference_matmul_perf_smoke(
    gpu_session: InferenceSession,
) -> None:
    torch.manual_seed(13)
    m = 512
    n = 512
    k = 2048
    sf_vector_size = 16
    packed_k = k // 2
    a_sf_shape = _scale_shape(m, k, sf_vector_size)
    b_sf_shape = _scale_shape(n, k, sf_vector_size)
    device = torch.device("cuda")

    a = torch.randint(0, 256, (m, packed_k), dtype=torch.uint8, device=device)
    b = torch.randint(0, 256, (n, packed_k), dtype=torch.uint8, device=device)
    a_scales_fp8 = (
        torch.rand(a_sf_shape, dtype=torch.float32, device=device) + 0.1
    ).to(torch.float8_e4m3fn)
    b_scales_fp8 = (
        torch.rand(b_sf_shape, dtype=torch.float32, device=device) + 0.1
    ).to(torch.float8_e4m3fn)
    tensor_sf = torch.tensor(0.25, dtype=torch.float32, device=device)

    a_deq = _unpack_reference(a, a_scales_fp8.to(torch.float32), sf_vector_size)
    b_deq = _unpack_reference(b, b_scales_fp8.to(torch.float32), sf_vector_size)

    ref_model = _build_reference_matmul_model(
        gpu_session,
        m,
        n,
        packed_k,
        a_sf_shape,
        b_sf_shape,
        sf_vector_size,
        DType.float8_e4m3fn,
    )
    baseline_model = _build_baseline_matmul_model(gpu_session, m, n, k)

    dev = gpu_session.devices[0]
    ref_ms = _run_timed(
        ref_model,
        (
            Buffer.from_dlpack(a),
            Buffer.from_dlpack(b),
            _to_max_scale_buffer(a_scales_fp8, DType.float8_e4m3fn),
            _to_max_scale_buffer(b_scales_fp8, DType.float8_e4m3fn),
            Buffer.from_dlpack(tensor_sf),
        ),
        dev,
        warmup=8,
        iters=25,
    )
    baseline_ms = _run_timed(
        baseline_model,
        (
            Buffer.from_dlpack(a_deq),
            Buffer.from_dlpack(b_deq),
            Buffer.from_dlpack(tensor_sf),
        ),
        dev,
        warmup=8,
        iters=25,
    )

    slowdown = ref_ms / baseline_ms
    print(
        "MXFP4 reference perf (ms): "
        f"ref={ref_ms:.3f}, baseline_bf16={baseline_ms:.3f}, slowdown={slowdown:.2f}x"
    )

    assert ref_ms > 0.0
    assert baseline_ms > 0.0
