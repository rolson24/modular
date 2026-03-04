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

import logging
import os
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from math import prod
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from max.diagnostics.gpu import GPUDiagContext
from max.driver import Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType
from max.graph.weights import Weights, WeightsAdapter
from max.nn.comm import Signals
from max.nn.kv_cache import (
    KVCacheInputs,
    KVCacheInputsSequence,
    KVCacheParams,
)
from max.nn.transformer import ReturnLogits
from max.pipelines.core import TextContext
from max.pipelines.lib import (
    AlwaysSignalBuffersMixin,
    CompilationTimer,
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModelWithKVCache,
)
from transformers import AutoConfig

from .gpt_oss import GptOss
from .model_config import GptOssConfig

logger = logging.getLogger("max.pipelines")

_MEMORY_DEBUG_ENV_VAR = "MAX_GPT_OSS_MEM_DEBUG"


def _memory_debug_enabled() -> bool:
    value = os.environ.get(_MEMORY_DEBUG_ENV_VAR, "")
    return value.lower() in {"1", "true", "yes", "on"}


def _format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} PiB"


def _get_process_rss_bytes() -> int | None:
    # Linux-specific current RSS probe.
    try:
        with open("/proc/self/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    fields = line.split()
                    return int(fields[1]) * 1024
    except Exception:
        return None
    return None


def _estimate_state_dict_bytes(state_dict: dict[str, Any]) -> int | None:
    total_bytes = 0
    num_estimated = 0
    for value in state_dict.values():
        nbytes = _tensor_nbytes(value)
        if nbytes is None:
            continue
        total_bytes += nbytes
        num_estimated += 1
    if num_estimated == 0:
        return None
    return total_bytes


def _tensor_nbytes(value: Any) -> int | None:
    dtype = getattr(value, "dtype", None)
    shape = getattr(value, "shape", None)
    size_in_bytes = getattr(dtype, "size_in_bytes", None)
    if size_in_bytes is None or shape is None:
        return None
    try:
        return int(prod(int(dim) for dim in shape)) * int(size_in_bytes)
    except Exception:
        return None


def _log_state_dict_summary(
    stage: str, state_dict: dict[str, Any], top_k: int = 8
) -> None:
    if not _memory_debug_enabled():
        return

    dtype_totals: dict[str, int] = {}
    largest: list[tuple[str, int, str, str]] = []
    estimated_entries = 0
    for name, value in state_dict.items():
        nbytes = _tensor_nbytes(value)
        if nbytes is None:
            continue
        estimated_entries += 1
        dtype = getattr(value, "dtype", None)
        dtype_name = str(dtype) if dtype is not None else "unknown"
        shape = getattr(value, "shape", None)
        shape_name = str(shape) if shape is not None else "unknown"
        dtype_totals[dtype_name] = dtype_totals.get(dtype_name, 0) + nbytes
        largest.append((name, nbytes, dtype_name, shape_name))

    if estimated_entries == 0:
        logger.info(
            "[gpt_oss_mem] %s state_dict_summary | no estimable entries",
            stage,
        )
        return

    dtype_parts = ", ".join(
        f"{dtype}={_format_bytes(total)}"
        for dtype, total in sorted(
            dtype_totals.items(), key=lambda item: item[1], reverse=True
        )
    )
    logger.info(
        "[gpt_oss_mem] %s state_dict_summary | entries=%d | by_dtype={%s}",
        stage,
        estimated_entries,
        dtype_parts,
    )

    for idx, (name, nbytes, dtype_name, shape_name) in enumerate(
        sorted(largest, key=lambda item: item[1], reverse=True)[:top_k],
        start=1,
    ):
        logger.info(
            "[gpt_oss_mem] %s state_dict_top[%d] | %s | %s | %s | %s",
            stage,
            idx,
            name,
            _format_bytes(nbytes),
            dtype_name,
            shape_name,
        )


def _log_graph_weight_contract(
    stage: str, graph: Any, state_dict: dict[str, Any], top_k: int = 8
) -> None:
    if not _memory_debug_enabled():
        return

    graph_weights = getattr(graph, "_weights", None)
    if not isinstance(graph_weights, dict):
        logger.info(
            "[gpt_oss_mem] %s graph_weight_contract | unavailable",
            stage,
        )
        return

    expected_total = 0
    provided_total = 0
    expected_entries = 0
    provided_entries = 0
    expected_largest: list[tuple[str, int, str, str]] = []
    mismatches: list[str] = []
    missing = 0
    expected_by_device: dict[str, int] = {}

    for name, graph_weight in graph_weights.items():
        value = getattr(graph_weight, "value", None)
        exp_nbytes = _tensor_nbytes(value) if value is not None else None
        exp_dtype = str(getattr(value, "dtype", "unknown"))
        exp_shape = str(getattr(value, "shape", "unknown"))
        exp_device = str(getattr(value, "device", "unknown"))

        if exp_nbytes is not None:
            expected_total += exp_nbytes
            expected_entries += 1
            expected_largest.append((name, exp_nbytes, exp_dtype, exp_shape))
            expected_by_device[exp_device] = (
                expected_by_device.get(exp_device, 0) + exp_nbytes
            )

        provided = state_dict.get(name)
        if provided is None:
            missing += 1
            continue

        prov_nbytes = _tensor_nbytes(provided)
        prov_dtype = str(getattr(provided, "dtype", "unknown"))
        prov_shape = str(getattr(provided, "shape", "unknown"))
        if prov_nbytes is not None:
            provided_total += prov_nbytes
            provided_entries += 1

        if exp_dtype != prov_dtype or exp_shape != prov_shape:
            mismatches.append(
                f"{name}: expected({exp_dtype}, {exp_shape}) "
                f"provided({prov_dtype}, {prov_shape})"
            )

    logger.info(
        "[gpt_oss_mem] %s graph_weight_contract | graph_entries=%d "
        "| expected_total=%s | provided_entries=%d | provided_total=%s "
        "| missing=%d | mismatches=%d",
        stage,
        len(graph_weights),
        _format_bytes(expected_total),
        provided_entries,
        _format_bytes(provided_total),
        missing,
        len(mismatches),
    )
    if expected_by_device:
        device_parts = ", ".join(
            f"{device}={_format_bytes(total)}"
            for device, total in sorted(
                expected_by_device.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )
        logger.info(
            "[gpt_oss_mem] %s graph_weight_devices | %s",
            stage,
            device_parts,
        )

    for idx, (name, nbytes, dtype_name, shape_name) in enumerate(
        sorted(expected_largest, key=lambda item: item[1], reverse=True)[:top_k],
        start=1,
    ):
        logger.info(
            "[gpt_oss_mem] %s graph_weight_top[%d] | %s | %s | %s | %s",
            stage,
            idx,
            name,
            _format_bytes(nbytes),
            dtype_name,
            shape_name,
        )

    for mismatch in mismatches[:top_k]:
        logger.info(
            "[gpt_oss_mem] %s graph_weight_mismatch | %s",
            stage,
            mismatch,
        )


def _estimate_weights_source_bytes(weights: Any) -> int | None:
    filepaths = getattr(weights, "_filepaths", None)
    if not filepaths:
        return None

    total_bytes = 0
    seen: set[str] = set()
    for path_like in filepaths:
        path = os.fspath(path_like)
        if path in seen:
            continue
        seen.add(path)
        try:
            total_bytes += os.path.getsize(path)
        except OSError:
            continue
    return total_bytes if total_bytes > 0 else None


def _log_allocator_settings(
    kv_cache_config: KVCacheConfig, pipeline_config: PipelineConfig
) -> None:
    if not _memory_debug_enabled():
        return

    env_keys = [
        "MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT",
        "MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE",
        "MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_CHUNK_PERCENT",
        "MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_ONLY",
    ]
    env_parts = [
        f"{key}={os.environ[key]}" for key in env_keys if key in os.environ
    ]

    parts = [
        f"kv_device_memory_utilization={kv_cache_config.device_memory_utilization:.3f}",
        f"kv_available_cache_memory={_format_bytes(kv_cache_config._available_cache_memory or 0)}",
        f"max_batch_size={pipeline_config.max_batch_size}",
        f"max_length={pipeline_config.model.max_length}",
    ]
    parts.extend(env_parts)
    logger.info("[gpt_oss_mem] allocator | %s", " | ".join(parts))


def _query_nvidia_memory() -> str | None:
    pid = os.getpid()
    try:
        with GPUDiagContext() as diag_ctx:
            stats = diag_ctx.get_stats()
    except Exception:
        stats = {}

    gpu_stats: list[str] = []
    for gpu_id, gpu_stats_obj in stats.items():
        used_bytes = gpu_stats_obj.memory.used_bytes
        total_bytes = gpu_stats_obj.memory.total_bytes
        free_bytes = gpu_stats_obj.memory.free_bytes
        reserved_bytes = gpu_stats_obj.memory.reserved_bytes
        formatted = (
            f"{gpu_id}=used:{_format_bytes(used_bytes)}/{_format_bytes(total_bytes)}"
            f",free:{_format_bytes(free_bytes)}"
        )
        if reserved_bytes is not None:
            formatted += f",reserved:{_format_bytes(reserved_bytes)}"
        gpu_stats.append(formatted)

    process_mib = 0.0
    try:
        proc_result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=2.0,
        )
    except Exception:
        proc_result = None

    if proc_result is not None:
        for line in proc_result.stdout.splitlines():
            values = [part.strip() for part in line.split(",")]
            if len(values) != 2:
                continue
            pid_str, used_str = values
            try:
                if int(pid_str) != pid:
                    continue
                process_mib += float(used_str)
            except ValueError:
                continue
    if process_mib > 0.0:
        gpu_stats.append(f"pid_compute_mem={process_mib / 1024.0:.2f} GiB")

    if not gpu_stats:
        return None
    return ", ".join(gpu_stats)


def _log_memory_snapshot(
    stage: str, state_dict: dict[str, Any] | None = None
) -> None:
    if not _memory_debug_enabled():
        return

    rss_bytes = _get_process_rss_bytes()
    gpu_usage = _query_nvidia_memory()

    extras: list[str] = []
    if rss_bytes is not None:
        extras.append(f"rss={_format_bytes(rss_bytes)}")
    if gpu_usage:
        extras.append(gpu_usage)
    if state_dict is not None:
        extras.append(f"state_dict_entries={len(state_dict)}")
        estimated_state_dict_bytes = _estimate_state_dict_bytes(state_dict)
        if estimated_state_dict_bytes is not None:
            extras.append(
                "state_dict_est="
                + _format_bytes(estimated_state_dict_bytes)
            )

    if extras:
        logger.info("[gpt_oss_mem] %s | %s", stage, " | ".join(extras))
    else:
        logger.info("[gpt_oss_mem] %s", stage)


@dataclass
class GptOssInputs(ModelInputs):
    """A class representing inputs for the GPT OSS model.

    This class encapsulates the input tensors required for the GPT OSS model
    execution.
    """

    tokens: npt.NDArray[np.integer[Any]] | Buffer
    """Tensor containing the input token IDs."""

    input_row_offsets: npt.NDArray[np.integer[Any]] | Buffer | list[Buffer]
    """Tensor containing the offsets for each row in the ragged input sequence,
    or the attention mask for the padded input sequence. For distributed execution,
    this can be a list of tensors, one per device."""

    signal_buffers: list[Buffer]
    """Device buffers used for synchronization in communication collectives."""

    return_n_logits: Buffer
    """Number of logits to return."""


class GptOssModel(
    AlwaysSignalBuffersMixin, PipelineModelWithKVCache[TextContext]
):
    """A GPT OSS pipeline model for text generation.

    This class integrates the GPT OSS architecture with the MAX Engine pipeline
    infrastructure, handling model loading, KV cache management, and input preparation
    for inference.
    """

    model: Model
    """The compiled and initialized MAX Engine model ready for inference."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        adapter: WeightsAdapter | None = None,
        return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN,
    ) -> None:
        """
        Args:
            pipeline_config: The configuration settings for the entire pipeline.
            session: The MAX Engine inference session managing the runtime.
            devices: A list of MAX Engine devices (:obj:`max.driver.Device`) to
                run the model on.
            kv_cache_config: Configuration settings for the Key-Value cache
                (:obj:`max.pipelines.max_config.KVCacheConfig`).
            weights: The model weights (:obj:`max.graph.weights.Weights`).
            adapter: An optional adapter to modify weights before loading
                (:obj:`max.graph.weights.WeightsAdapter`).
            return_logits: The number of top logits to return from the model
                execution.
        """
        _log_memory_snapshot("__init__:before_super")
        super().__init__(
            pipeline_config,
            session,
            devices,
            kv_cache_config,
            weights,
            adapter,
            return_logits,
        )
        _log_memory_snapshot("__init__:after_super")
        source_bytes = _estimate_weights_source_bytes(self.weights)
        if source_bytes is not None:
            logger.info(
                "[gpt_oss_mem] weights_source_on_disk=%s",
                _format_bytes(source_bytes),
            )
        if _memory_debug_enabled():
            try:
                from max.nn.moe import stacked_moe as _stacked_moe_module

                logger.info(
                    "[gpt_oss_mem] stacked_moe_module_file=%s",
                    _stacked_moe_module.__file__,
                )
            except Exception:
                pass

        self.model = self.load_model(session)

    @classmethod
    def estimate_activation_memory(
        cls, pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        del huggingface_config  # Unused.

        # FIXME GEX-3248: This is a workaround for a MemoryManager fragmentation
        # issue. In #77700 we swapped the order of model weight loading and kv
        # cache loading. This affected memory fragmentation and led to CUDA OOM
        # when running `br smoke-test -- unsloth/gpt-oss-20b-bf16` on 1xH100.
        # We reduce the kv cache size slightly to avoid this.
        #
        # MXFP4 on single consumer Blackwell (e.g. RTX 5090) has materially
        # lower static weight footprint than BF16. Using the BF16 reserve here
        # can incorrectly reject valid configurations during preflight memory
        # estimation before runtime has a chance to compile/load.
        if pipeline_config.model.quantization_encoding == "float4_e2m1fnx2":
            return 2 * 1024 * 1024 * 1024  # 2 GiB
        return 6 * 1024 * 1024 * 1024  # 6 GiB

    @staticmethod
    def calculate_max_seq_len(
        pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        """Calculates the maximum sequence length for the GPT OSS model.

        Uses the `max_length` from the :obj:`max.pipelines.config.PipelineConfig`
        if provided, otherwise falls back to the `max_position_embeddings` from
        the HuggingFace configuration's text config.

        Args:
            pipeline_config: The MAX Engine pipeline configuration.
            huggingface_config: The HuggingFace model configuration object
                (:obj:`transformers.AutoConfig`).

        Returns:
            The calculated maximum sequence length.
        """
        max_seq_len = pipeline_config.model.max_length
        if max_seq_len:
            return max_seq_len
        return huggingface_config.max_position_embeddings

    @classmethod
    def get_kv_params(
        cls,
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        """Gets the parameters required to configure the KV cache for Gemma 3.

        Delegates to the :obj:`GptOssConfig.construct_kv_params` static method.

        Args:
            huggingface_config: The HuggingFace model configuration object
                (:obj:`transformers.AutoConfig`).
            pipeline_config: The MAX Engine pipeline configuration.
            devices: The list of devices the model will run on.
            kv_cache_config: The MAX Engine KV cache configuration settings
                (:obj:`max.pipelines.max_config.KVCacheConfig`).
            cache_dtype: The desired data type for the KV cache
                (:obj:`max.dtype.DType`).

        Returns:
            The configured :obj:`max.pipelines.kv_cache.KVCacheParams` object.
        """
        return GptOssConfig.construct_kv_params(
            huggingface_config,
            pipeline_config,
            devices,
            kv_cache_config,
            cache_dtype,
        )

    def load_model(self, session: InferenceSession) -> Model:
        """Loads the compiled GPT OSS model into the MAX Engine session.

        Args:
            session: The MAX Engine inference session.

        Returns:
            The loaded MAX Engine model object.
        """
        assert self.pipeline_config.max_batch_size, (
            "Expected max_batch_size to be set"
        )
        _log_allocator_settings(self.kv_cache_config, self.pipeline_config)
        _log_memory_snapshot("load_model:before_prealloc")
        self._input_row_offsets_prealloc = Buffer.from_numpy(
            np.arange(self.pipeline_config.max_batch_size + 1, dtype=np.uint32)
        ).to(self.devices[0])
        _log_memory_snapshot("load_model:after_prealloc")

        timer = CompilationTimer("model")
        _log_memory_snapshot("load_model:start")
        graph = self._build_graph()
        timer.mark_build_complete()
        _log_memory_snapshot("load_model:after_build_graph")
        _log_graph_weight_contract(
            "load_model:before_session_load",
            graph,
            self.state_dict,
        )
        try:
            model = session.load(graph, weights_registry=self.state_dict)
        except Exception:
            _log_memory_snapshot("load_model:session_load_exception")
            raise
        _log_memory_snapshot("load_model:after_session_load")
        timer.done()

        return model

    # For text-only models, we should be using all the weights.
    _strict_state_dict_loading = True

    def _build_graph(self):  # noqa: ANN202
        device0 = self.devices[0]
        device_ref = DeviceRef(device0.label, device0.id)
        tokens_type = TensorType(
            DType.int64, shape=["total_seq_len"], device=device_ref
        )
        # NOTE: input_row_offsets_len should be batch_size + 1.
        # Create input_row_offsets_type for each device
        input_row_offsets_types = [
            TensorType(
                DType.uint32,
                shape=["input_row_offsets_len"],
                device=DeviceRef(device.label, device.id),
            )
            for device in self.devices
        ]
        return_n_logits_type = TensorType(
            DType.int64, shape=["return_n_logits"], device=DeviceRef.CPU()
        )
        signals = Signals(
            devices=(DeviceRef(d.label, d.id) for d in self.devices)
        )

        huggingface_config = self.huggingface_config
        _log_memory_snapshot("_build_graph:before_state_dict")
        if self.adapter:
            state_dict = self.adapter(
                dict(self.weights.items()),
                huggingface_config=huggingface_config,
                pipeline_config=self.pipeline_config,
            )
        else:
            state_dict = {
                key: value.data() for key, value in self.weights.items()
            }
        _log_memory_snapshot("_build_graph:after_state_dict", state_dict)
        _log_state_dict_summary("_build_graph:after_state_dict", state_dict)
        model_config = GptOssConfig.initialize(self.pipeline_config)
        model_config.finalize(
            huggingface_config=huggingface_config,
            state_dict=state_dict,
            return_logits=self.return_logits,
        )
        _log_memory_snapshot("_build_graph:after_config_finalize", state_dict)
        nn_model = GptOss(model_config)
        nn_model.load_state_dict(
            state_dict,
            weight_alignment=1,
            strict=self._strict_state_dict_loading,
        )
        _log_memory_snapshot("_build_graph:after_nn_load_state_dict")
        self.state_dict = nn_model.state_dict(auto_initialize=False)
        _log_memory_snapshot(
            "_build_graph:after_materialize_nn_state_dict", self.state_dict
        )
        _log_state_dict_summary(
            "_build_graph:after_materialize_nn_state_dict",
            self.state_dict,
        )

        # Create signal types for distributed communication
        signals = Signals(
            devices=(DeviceRef(d.label, d.id) for d in self.devices)
        )

        kv_inputs = self.kv_params.get_symbolic_inputs()
        flattened_kv_types = kv_inputs.flatten()

        with Graph(
            getattr(self.huggingface_config, "model_type", "GptOss"),
            input_types=[
                tokens_type,
                return_n_logits_type,
                *input_row_offsets_types,
                *signals.input_types(),
                *flattened_kv_types,
            ],
        ) as graph:
            # Unpack inputs following InternVL pattern
            tokens, return_n_logits, *variadic_args = graph.inputs

            # Extract input_row_offsets (one per device)
            input_row_offsets = [
                v.tensor for v in variadic_args[: len(self.devices)]
            ]
            variadic_args = variadic_args[len(self.devices) :]

            # Extract signal buffers (one per device)
            signal_buffers = [
                v.buffer for v in variadic_args[: len(self.devices)]
            ]
            variadic_args = variadic_args[len(self.devices) :]

            # Extract KV cache inputs
            kv_cache = self._unflatten_kv_inputs(variadic_args)

            outputs = nn_model(
                tokens=tokens.tensor,
                signal_buffers=signal_buffers,
                kv_cache_inputs_per_dev=kv_cache,
                return_n_logits=return_n_logits.tensor,
                input_row_offsets=input_row_offsets,
            )
            graph.output(*outputs)
        return graph

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        """Executes the GPT OSS model with the prepared inputs.

        Args:
            model_inputs: The prepared inputs for the model execution, typically including
                token IDs, attention masks/offsets, and KV cache inputs.

        Returns:
            An object containing the output logits from the model execution.
        """
        model_inputs = cast(GptOssInputs, model_inputs)
        curr_kv_cache_inputs = model_inputs.kv_cache_inputs or ()

        # Check if input_row_offsets is a list or a single tensor
        if isinstance(model_inputs.input_row_offsets, list):
            input_row_offsets_list = model_inputs.input_row_offsets
        else:
            # For backward compatibility, distribute the single tensor to all devices
            if isinstance(model_inputs.input_row_offsets, np.ndarray):
                # Convert numpy array to tensor first
                tensor = Buffer.from_numpy(model_inputs.input_row_offsets)
                input_row_offsets_list = [
                    tensor.to(device) for device in self.devices
                ]
            else:
                # Already a tensor
                input_row_offsets_list = [
                    model_inputs.input_row_offsets.to(device)
                    for device in self.devices
                ]

        model_outputs = self.model.execute(
            model_inputs.tokens,
            model_inputs.return_n_logits,
            *input_row_offsets_list,
            *model_inputs.signal_buffers,
            *curr_kv_cache_inputs,
        )
        if len(model_outputs) == 3:
            return ModelOutputs(
                logits=cast(Buffer, model_outputs[1]),
                next_token_logits=cast(Buffer, model_outputs[0]),
                logit_offsets=cast(Buffer, model_outputs[2]),
            )
        else:
            return ModelOutputs(
                logits=cast(Buffer, model_outputs[0]),
                next_token_logits=cast(Buffer, model_outputs[0]),
            )

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputs | None = None,
        return_n_logits: int = 1,
    ) -> ModelInputs:
        """Prepares the initial inputs for the first execution pass of the GPT OSS model.

        Args:
            context_batch: A sequence of :obj:`TextContext` objects representing
                the input prompts.
            kv_cache_inputs: Optional inputs required by the KV cache manager.

        Returns:
            The prepared :obj:`ModelInputs` object for the initial execution step.
        """
        if len(replica_batches) > 1:
            raise ValueError("Model does not support DP>1")

        context_batch = replica_batches[0]
        assert kv_cache_inputs is not None
        kv_cache_inputs = cast(KVCacheInputsSequence, kv_cache_inputs)

        # This needs to be replaced with actual input preparation
        # Get input_row_offsets: start and end position of each batch in the
        # combined total_seq_len dimension.
        input_row_offsets = np.cumsum(
            [0] + [ctx.tokens.active_length for ctx in context_batch],
            dtype=np.uint32,
        )

        # Create a ragged token vector of length: sum(len(t) for t in tokens).
        tokens = np.concatenate([ctx.tokens.active for ctx in context_batch])

        # Create input_row_offsets for each device
        input_row_offsets_tensors = [
            Buffer.from_numpy(input_row_offsets).to(device)
            for device in self.devices
        ]

        return GptOssInputs(
            tokens=Buffer.from_numpy(tokens).to(self.devices[0]),
            input_row_offsets=input_row_offsets_tensors,
            return_n_logits=Buffer.from_numpy(
                np.array([return_n_logits], dtype=np.int64)
            ),
            signal_buffers=self.signal_buffers,
            kv_cache_inputs=kv_cache_inputs,
        )

    def prepare_next_token_inputs(
        self, next_tokens: Buffer, prev_model_inputs: ModelInputs
    ) -> ModelInputs:
        """Prepares the inputs for subsequent execution steps in a multi-step generation.

        Args:
            next_tokens: The tensor containing the token IDs generated in the previous step.
            prev_model_inputs: The :obj:`ModelInputs` used in the previous execution step.

        Returns:
            The prepared :obj:`ModelInputs` object for the next execution step.
        """
        prev_model_inputs = cast(GptOssInputs, prev_model_inputs)

        row_offsets_size = prev_model_inputs.input_row_offsets[0].shape[0]

        next_row_offsets = [
            self._input_row_offsets_prealloc[:row_offsets_size].to(device)
            for device in self.devices
        ]

        return GptOssInputs(
            tokens=next_tokens,
            input_row_offsets=next_row_offsets,
            return_n_logits=prev_model_inputs.return_n_logits,
            signal_buffers=self.signal_buffers,
            kv_cache_inputs=prev_model_inputs.kv_cache_inputs,
        )
