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

import numpy as np
from max.dtype import DType
from max.graph.weights import WeightData
from max.pipelines.architectures.gpt_oss.weight_adapters import (
    convert_safetensor_state_dict,
)


class _MockWeight:
    def __init__(self, array: np.ndarray, name: str):
        self._weight_data = WeightData.from_numpy(array, name)

    def data(self) -> WeightData:
        return self._weight_data


def test_convert_safetensor_state_dict_mxfp4_moe() -> None:
    state_dict = {
        "model.layers.0.mlp.experts.gate_up_proj_blocks": _MockWeight(
            np.zeros((2, 8, 3, 16), dtype=np.uint8),
            "model.layers.0.mlp.experts.gate_up_proj_blocks",
        ),
        "model.layers.0.mlp.experts.gate_up_proj_scales": _MockWeight(
            np.full((2, 8, 3), 127, dtype=np.uint8),
            "model.layers.0.mlp.experts.gate_up_proj_scales",
        ),
        "model.layers.0.mlp.experts.down_proj_blocks": _MockWeight(
            np.zeros((2, 4, 5, 16), dtype=np.uint8),
            "model.layers.0.mlp.experts.down_proj_blocks",
        ),
        "model.layers.0.mlp.experts.down_proj_scales": _MockWeight(
            np.full((2, 4, 5), 127, dtype=np.uint8),
            "model.layers.0.mlp.experts.down_proj_scales",
        ),
        "model.layers.0.mlp.router.weight": _MockWeight(
            np.zeros((2, 4), dtype=np.float32),
            "model.layers.0.mlp.router.weight",
        ),
    }

    converted = convert_safetensor_state_dict(state_dict)  # type: ignore[arg-type]

    assert (
        tuple(converted["language_model.layers.0.mlp.experts.gate_up_proj"].shape)
        == (2, 8, 48)
    )
    assert (
        tuple(converted["language_model.layers.0.mlp.experts.down_proj"].shape)
        == (2, 4, 80)
    )
    assert (
        converted["language_model.layers.0.mlp.experts.gate_up_proj_scale"].dtype
        == DType.float8_e8m0fnu
    )
    assert (
        converted["language_model.layers.0.mlp.experts.down_proj_scale"].dtype
        == DType.float8_e8m0fnu
    )
    assert "language_model.layers.0.mlp.gate.gate_score.weight" in converted
