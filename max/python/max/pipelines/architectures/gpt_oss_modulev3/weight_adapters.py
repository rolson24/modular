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

from max.driver import Buffer
from max.dtype import DType
from max.graph.weights import WeightData, Weights

GPT_OSS_SAFETENSOR_MAP: dict[str, str] = {
    "model.embed_tokens.": "language_model.embed_tokens.",
    "model.norm.": "language_model.norm.",
    "lm_head.": "language_model.lm_head.",
    "model.layers.": "language_model.layers.",
    # OpenAI GPT-OSS MXFP4 MoE checkpoint names.
    ".mlp.experts.gate_up_proj_blocks": ".mlp._experts_gate_up_proj_weight",
    ".mlp.experts.gate_up_proj_scales": ".mlp._experts_gate_up_proj_weight_scale",
    ".mlp.experts.down_proj_blocks": ".mlp._experts_down_proj_weight",
    ".mlp.experts.down_proj_scales": ".mlp._experts_down_proj_weight_scale",
    # MoE weight mappings
    ".mlp.router": ".mlp.gate.gate_score",
    "experts.gate_up_proj_bias": "_experts_gate_up_proj_bias",
    "experts.down_proj_bias": "_experts_down_proj_bias",
    # The following weights must be listed after the bias weights, because
    # they share the same prefix.
    "experts.gate_up_proj": "_experts_gate_up_proj_weight",
    "experts.down_proj": "_experts_down_proj_weight",
}


def _flatten_block_tail(weight_data: WeightData) -> WeightData:
    """Converts block-packed [*, k_groups, 16] tensors into [*, packed_k]."""
    if len(weight_data.shape) != 4:
        return weight_data
    buf = Buffer.from_dlpack(weight_data.data)
    if buf.shape[-1] != 16:
        return weight_data

    flattened = buf.view(
        dtype=buf.dtype,
        shape=(buf.shape[0], buf.shape[1], buf.shape[2] * buf.shape[3]),
    )
    return WeightData(
        data=flattened,
        name=weight_data.name,
        dtype=weight_data.dtype,
        shape=weight_data.shape.__class__(flattened.shape),
        quantization_encoding=weight_data.quantization_encoding,
    )


def _reinterpret_ue8m0(weight_data: WeightData) -> WeightData:
    """Reinterprets uint8 UE8M0 scale bytes as float8_e8m0fnu."""
    if weight_data.dtype != DType.uint8:
        return weight_data

    buf = Buffer.from_dlpack(weight_data.data)
    viewed = buf.view(dtype=DType.float8_e8m0fnu, shape=buf.shape)
    return WeightData(
        data=viewed,
        name=weight_data.name,
        dtype=DType.float8_e8m0fnu,
        shape=weight_data.shape,
        quantization_encoding=weight_data.quantization_encoding,
    )


def convert_safetensor_state_dict(
    state_dict: dict[str, Weights], **kwargs
) -> dict[str, WeightData]:
    """Convert safetensor state dict to MAX format.

    Args:
        state_dict: Dictionary of weight tensors

    Returns:
        Dictionary of converted weight data
    """

    # Now remap all weight names from HuggingFace to MAX format
    new_state_dict: dict[str, WeightData] = {}

    for weight_name, value in state_dict.items():
        weight_data = value.data()
        if weight_name.endswith("_blocks") and ".mlp.experts." in weight_name:
            weight_data = _flatten_block_tail(weight_data)
        elif (
            weight_name.endswith("_scales")
            and ".mlp.experts." in weight_name
        ):
            weight_data = _reinterpret_ue8m0(weight_data)

        max_name: str = weight_name
        for before, after in GPT_OSS_SAFETENSOR_MAP.items():
            max_name = max_name.replace(before, after)
        new_state_dict[max_name] = weight_data

    return new_state_dict
