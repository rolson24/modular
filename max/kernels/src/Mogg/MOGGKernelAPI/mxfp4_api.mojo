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

import compiler_internal as compiler

from linalg.fp4_utils import cast_uint_to_fp4e2m1
from runtime.asyncrt import DeviceContextPtr
from std.algorithm import elementwise
from std.math import ceildiv
from std.utils import IndexList
from tensor import InputTensor, OutputTensor


@compiler.register("mo.mxfp4.unpack")
struct Struct_mxfp4_unpack:
    @staticmethod
    fn execute[
        scales_type: DType,
        //,
        SF_VECTOR_SIZE: Int,
        target: StaticString,
    ](
        output: OutputTensor[dtype = DType.bfloat16, rank=2],
        packed: InputTensor[dtype = DType.uint8, rank=2],
        scales: InputTensor[dtype=scales_type, rank=5],
        context: DeviceContextPtr,
    ) raises:
        comptime assert SF_VECTOR_SIZE in (16, 32), (
            "SF_VECTOR_SIZE must be 16 (NVFP4) or 32 (MXFP4)"
        )
        comptime assert scales_type in (
            DType.float8_e4m3fn,
            DType.float8_e8m0fnu,
        ), (
            "scales_type must be float8_e4m3fn (NVFP4) or float8_e8m0fnu"
            " (MXFP4)"
        )
        comptime assert (
            (SF_VECTOR_SIZE == 16 and scales_type == DType.float8_e4m3fn)
            or (SF_VECTOR_SIZE == 32 and scales_type == DType.float8_e8m0fnu)
        ), (
            "SF_VECTOR_SIZE/scales_type mismatch: expected"
            " (16, float8_e4m3fn) for NVFP4 or"
            " (32, float8_e8m0fnu) for MXFP4"
        )

        comptime SF_ATOM_M0 = 32
        comptime SF_ATOM_K = 4
        comptime SF_MN_GROUP_SIZE = SF_ATOM_M0 * 4
        comptime SF_K_GROUP_SIZE = SF_ATOM_K * SF_VECTOR_SIZE

        if output.dim_size(0) != packed.dim_size(0):
            raise Error(
                "output.rows must match packed.rows, got ",
                output.dim_size(0),
                " and ",
                packed.dim_size(0),
            )

        if output.dim_size(1) != packed.dim_size(1) * 2:
            raise Error(
                "output.cols must be packed.cols * 2, got ",
                output.dim_size(1),
                " and ",
                packed.dim_size(1) * 2,
            )

        var expected_scales_rows = ceildiv(output.dim_size(0), SF_MN_GROUP_SIZE)
        var expected_scales_cols = ceildiv(output.dim_size(1), SF_K_GROUP_SIZE)
        if (
            scales.dim_size(0) != expected_scales_rows
            or scales.dim_size(1) != expected_scales_cols
            or scales.dim_size(2) != SF_ATOM_M0
            or scales.dim_size(3) != 4
            or scales.dim_size(4) != SF_ATOM_K
        ):
            raise Error(
                "Invalid scales shape for unpacked tensor shape ",
                output.shape(),
            )

        var output_tensor = output.to_layout_tensor()
        var packed_tensor = packed.to_layout_tensor()
        var scales_tensor = scales.to_layout_tensor()

        @__copy_capture(output_tensor, packed_tensor, scales_tensor)
        @parameter
        @always_inline
        fn unpack_fn[
            simd_width: Int, rank: Int, alignment: Int = 1
        ](index: IndexList[rank]):
            var idx = rebind[IndexList[2]](index)
            var row = idx[0]
            var col = idx[1]

            var packed_col = col // 2
            var packed_byte = rebind[Scalar[DType.uint8]](
                packed_tensor[row, packed_col]
            )
            var fp4_pair = cast_uint_to_fp4e2m1[
                in_dtype=DType.uint8,
                in_width=1,
                out_dtype=DType.float32,
                out_width=2,
            ](SIMD[DType.uint8, 1](packed_byte))

            var fp4_value = rebind[Scalar[DType.float32]](fp4_pair[col % 2])
            var scale = abs(
                rebind[Scalar[DType.float32]](
                    scales_tensor[
                        row // SF_MN_GROUP_SIZE,
                        col // SF_K_GROUP_SIZE,
                        row % SF_ATOM_M0,
                        (row % SF_MN_GROUP_SIZE) // SF_ATOM_M0,
                        (col // SF_VECTOR_SIZE) % SF_ATOM_K,
                    ].cast[DType.float32]()
                )
            )

            output_tensor[row, col] = (fp4_value * scale).cast[DType.bfloat16]()

        elementwise[
            func=unpack_fn,
            simd_width=1,
            target=target,
        ](output.shape(), context=context)
