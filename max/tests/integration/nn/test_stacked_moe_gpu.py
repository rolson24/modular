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

"""Minimal smoke test for StackedMoE layer."""

import torch
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.nn.float8_config import (
    Float8Config,
    Float8InputScaleSpec,
    Float8ScaleGranularity,
    Float8ScaleOrigin,
    Float8WeightScaleSpec,
)
from max.nn.moe import MoEGate, StackedMoE
from torch.utils.dlpack import from_dlpack

HIDDEN_DIM = 256
NUM_EXPERTS = 4
NUM_EXPERTS_PER_TOKEN = 2
MOE_DIM = 512
SEQ_LEN = 16
DTYPE = DType.bfloat16


def test_stacked_moe_basic() -> None:
    """Verify StackedMoE compiles and produces finite outputs."""
    torch.manual_seed(42)

    moe = StackedMoE(
        devices=[DeviceRef.GPU()],
        hidden_dim=HIDDEN_DIM,
        num_experts=NUM_EXPERTS,
        num_experts_per_token=NUM_EXPERTS_PER_TOKEN,
        moe_dim=MOE_DIM,
        gate_cls=MoEGate,
        dtype=DTYPE,
    )
    moe.load_state_dict(
        {
            "gate.gate_score.weight": torch.randn(
                NUM_EXPERTS, HIDDEN_DIM, dtype=torch.bfloat16
            ),
            "experts.gate_up_proj": torch.randn(
                NUM_EXPERTS, HIDDEN_DIM, 2 * MOE_DIM, dtype=torch.bfloat16
            )
            * 0.02,
            "experts.down_proj": torch.randn(
                NUM_EXPERTS, MOE_DIM, HIDDEN_DIM, dtype=torch.bfloat16
            )
            * 0.02,
        },
        strict=True,
    )

    device = Accelerator()
    session = InferenceSession(devices=[device])
    input_type = TensorType(
        DTYPE, [SEQ_LEN, HIDDEN_DIM], device=DeviceRef.GPU()
    )

    with Graph("StackedMoE_test", input_types=(input_type,)) as graph:
        x = graph.inputs[0]
        output = moe(x.tensor)
        graph.output(output)

    compiled = session.load(graph, weights_registry=moe.state_dict())

    hidden_states = torch.randn(
        SEQ_LEN, HIDDEN_DIM, dtype=torch.bfloat16, device="cuda"
    )
    result = compiled.execute(Buffer.from_dlpack(hidden_states).to(device))
    output_tensor = from_dlpack(result[0])

    assert output_tensor.shape == (SEQ_LEN, HIDDEN_DIM)
    assert torch.all(torch.isfinite(output_tensor))


def test_stacked_moe_mxfp4_basic() -> None:
    """Verify StackedMoE MXFP4 fallback path compiles and runs."""
    torch.manual_seed(7)

    hidden_dim = 256
    moe_dim = 512
    num_experts = 4
    packed_hidden = hidden_dim // 2
    packed_moe = moe_dim // 2
    hidden_k_groups = packed_hidden // 16
    moe_k_groups = packed_moe // 16

    mxfp4_config = Float8Config(
        input_scale=Float8InputScaleSpec(
            granularity=Float8ScaleGranularity.BLOCK,
            origin=Float8ScaleOrigin.STATIC,
            dtype=DType.float32,
            block_size=(1, 32),
        ),
        weight_scale=Float8WeightScaleSpec(
            granularity=Float8ScaleGranularity.BLOCK,
            dtype=DType.float8_e8m0fnu,
            block_size=(1, 16),
        ),
        mlp_in_float8={0},
        attn_qkv_in_float8=set(),
        quant_method="modelopt",
        quant_algo="MXFP4",
    )

    moe = StackedMoE(
        devices=[DeviceRef.GPU()],
        hidden_dim=hidden_dim,
        num_experts=num_experts,
        num_experts_per_token=NUM_EXPERTS_PER_TOKEN,
        moe_dim=moe_dim,
        gate_cls=MoEGate,
        dtype=DType.uint8,
        float8_config=mxfp4_config,
    )
    gate_up_scale = Buffer.from_dlpack(
        torch.full(
            (num_experts, 2 * moe_dim, hidden_k_groups),
            127,
            dtype=torch.uint8,
        )
    ).view(
        dtype=DType.float8_e8m0fnu,
        shape=(num_experts, 2 * moe_dim, hidden_k_groups),
    )
    down_scale = Buffer.from_dlpack(
        torch.full(
            (num_experts, hidden_dim, moe_k_groups),
            127,
            dtype=torch.uint8,
        )
    ).view(
        dtype=DType.float8_e8m0fnu,
        shape=(num_experts, hidden_dim, moe_k_groups),
    )
    moe.load_state_dict(
        {
            "gate.gate_score.weight": torch.randn(
                num_experts, hidden_dim, dtype=torch.bfloat16
            ),
            "experts.gate_up_proj": torch.randint(
                0,
                256,
                (num_experts, 2 * moe_dim, packed_hidden),
                dtype=torch.uint8,
            ),
            "experts.down_proj": torch.randint(
                0,
                256,
                (num_experts, hidden_dim, packed_moe),
                dtype=torch.uint8,
            ),
            "experts.gate_up_proj_scale": gate_up_scale,
            "experts.down_proj_scale": down_scale,
        },
        strict=True,
    )

    device = Accelerator()
    session = InferenceSession(devices=[device])
    input_type = TensorType(
        DType.bfloat16, [SEQ_LEN, hidden_dim], device=DeviceRef.GPU()
    )

    with Graph("StackedMoE_mxfp4_test", input_types=(input_type,)) as graph:
        x = graph.inputs[0]
        output = moe(x.tensor)
        graph.output(output)

    compiled = session.load(graph, weights_registry=moe.state_dict())

    hidden_states = torch.randn(
        SEQ_LEN, hidden_dim, dtype=torch.bfloat16, device="cuda"
    )
    result = compiled.execute(Buffer.from_dlpack(hidden_states).to(device))
    output_tensor = from_dlpack(result[0])

    assert output_tensor.shape == (SEQ_LEN, hidden_dim)
    assert torch.all(torch.isfinite(output_tensor))
