from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from torch import Tensor, nn


Geometry = Literal["coordinate", "row_column", "spectral"]
AccessPattern = Literal["dense", "token_lookup", "token_output", "coordinate"]
UpdateFrequency = Literal["every_step", "odd_step"]


@dataclass(frozen=True, slots=True)
class BlockDescriptor:
    label: str
    parameter_label: str
    shape: tuple[int, ...]
    matrix_shape: tuple[int, int] | None
    tensor_index: tuple[int | slice, ...]
    input_axis: int | None
    output_axis: int | None
    access_pattern: AccessPattern
    token_axis: int | None
    head_partition: tuple[int, ...] | None
    shared_weight: str | None
    update_frequency: UpdateFrequency
    storage_transposed: bool = False
    structure_class: str = "coordinate"


def select_block(tensor: Tensor, descriptor: BlockDescriptor) -> Tensor:
    if not descriptor.tensor_index:
        return tensor
    return tensor[descriptor.tensor_index]


def legal_geometries(descriptor: BlockDescriptor) -> tuple[Geometry, ...]:
    if descriptor.access_pattern == "coordinate" or descriptor.matrix_shape is None:
        return ("coordinate",)
    return ("coordinate", "row_column", "spectral")


def structure_prior(descriptor: BlockDescriptor, mode: Geometry) -> float:
    if descriptor.access_pattern == "dense" and mode == "spectral":
        return 0.10
    if descriptor.access_pattern in {"token_lookup", "token_output"} and mode == "row_column":
        return 0.10
    return 0.0


def _parameter_map(model: nn.Module) -> dict[str, nn.Parameter]:
    core = getattr(model, "_orig_mod", model)
    result: dict[str, nn.Parameter] = {}
    for name, parameter in core.named_parameters():
        label = getattr(parameter, "label", name.removesuffix(".weight"))
        if label in result:
            raise ValueError(f"duplicate parameter label: {label}")
        result[label] = parameter
    return result


def _matrix_descriptor(
    *,
    label: str,
    parameter_label: str,
    shape: tuple[int, ...],
    tensor_index: tuple[int | slice, ...],
    input_axis: int,
    output_axis: int,
    head_partition: tuple[int, ...] | None,
    storage_transposed: bool,
    structure_class: str,
) -> BlockDescriptor:
    return BlockDescriptor(
        label=label,
        parameter_label=parameter_label,
        shape=shape,
        matrix_shape=(shape[-2], shape[-1]),
        tensor_index=tensor_index,
        input_axis=input_axis,
        output_axis=output_axis,
        access_pattern="dense",
        token_axis=None,
        head_partition=head_partition,
        shared_weight=None,
        update_frequency="every_step",
        storage_transposed=storage_transposed,
        structure_class=structure_class,
    )


def build_gpt_descriptors(model: nn.Module) -> tuple[BlockDescriptor, ...]:
    core = getattr(model, "_orig_mod", model)
    parameters = _parameter_map(core)
    descriptors: list[BlockDescriptor] = []
    handled: set[str] = set()

    if "qk_bank" in parameters:
        parameter = parameters["qk_bank"]
        active = int(core._num_qk_groups)
        layer_count = int(core._num_attn_layers)
        groups_per_layer = active // layer_count
        pairs_per_kind = groups_per_layer // 2
        if active % layer_count or groups_per_layer % 2:
            raise ValueError("qk_bank active matrices do not form Q/K head pairs")
        for index in range(active):
            layer = index // groups_per_layer
            within_layer = index % groups_per_layer
            kind = "q" if within_layer < pairs_per_kind else "k"
            pair = within_layer % pairs_per_kind
            descriptors.append(
                _matrix_descriptor(
                    label=f"qk_bank.layer{layer}.{kind}.pair{pair}",
                    parameter_label="qk_bank",
                    shape=tuple(parameter[index].shape),
                    tensor_index=(index,),
                    input_axis=1,
                    output_axis=0,
                    head_partition=(layer, pair),
                    storage_transposed=False,
                    structure_class="attention",
                )
            )
        handled.add("qk_bank")

    if "vo_bank" in parameters:
        parameter = parameters["vo_bank"]
        active = int(
            getattr(core, "_num_vo_real", int(core._num_attn_layers) * 2)
        )
        if active % 2:
            raise ValueError("vo_bank active matrices must form V/O pairs")
        layer_count = active // 2
        if active > parameter.shape[0]:
            raise ValueError("vo_bank has fewer matrices than active V/O blocks")
        for index in range(active):
            layer, projection = divmod(index, 2)
            kind = "v" if projection == 0 else "o"
            descriptors.append(
                _matrix_descriptor(
                    label=f"vo_bank.layer{layer}.{kind}",
                    parameter_label="vo_bank",
                    shape=tuple(parameter[index].shape),
                    tensor_index=(index,),
                    input_axis=1,
                    output_axis=0,
                    head_partition=(layer,),
                    storage_transposed=False,
                    structure_class="attention",
                )
            )
        handled.add("vo_bank")

    if "mlp_bank" in parameters:
        parameter = parameters["mlp_bank"]
        layer_count = min(int(core.num_layers), int(parameter.shape[0]))
        for layer in range(layer_count):
            for projection, kind in enumerate(("up", "down")):
                transposed = projection == 1
                descriptors.append(
                    _matrix_descriptor(
                        label=f"mlp_bank.layer{layer}.{kind}",
                        parameter_label="mlp_bank",
                        shape=tuple(parameter[layer, projection].shape),
                        tensor_index=(layer, projection),
                        input_axis=0 if transposed else 1,
                        output_axis=1 if transposed else 0,
                        head_partition=None,
                        storage_transposed=transposed,
                        structure_class="mlp",
                    )
                )
        handled.add("mlp_bank")

    token_tables: dict[str, tuple[AccessPattern, int, bool, str, str | None]] = {
        "embed": ("token_lookup", 0, False, "embedding", "lm_head"),
        "bigram_embed": ("token_lookup", 0, False, "embedding", None),
        "value_embeds": ("token_lookup", 0, False, "embedding", None),
        "lm_head": ("token_output", 1, True, "lm_head", "embed"),
    }
    for label, (pattern, token_axis, transposed, structure_class, shared) in token_tables.items():
        if label not in parameters:
            continue
        parameter = parameters[label]
        if parameter.ndim != 2:
            raise ValueError(f"token block {label} must be rank 2")
        descriptors.append(
            BlockDescriptor(
                label=label,
                parameter_label=label,
                shape=tuple(parameter.shape),
                matrix_shape=(parameter.shape[0], parameter.shape[1]),
                tensor_index=(),
                input_axis=0 if transposed else 1,
                output_axis=1 if transposed else 0,
                access_pattern=pattern,
                token_axis=token_axis,
                head_partition=None,
                shared_weight=shared,
                update_frequency="odd_step",
                storage_transposed=transposed,
                structure_class=structure_class,
            )
        )
        handled.add(label)

    for label, parameter in parameters.items():
        if label in handled:
            continue
        descriptors.append(
            BlockDescriptor(
                label=label,
                parameter_label=label,
                shape=tuple(parameter.shape),
                matrix_shape=None,
                tensor_index=(),
                input_axis=None,
                output_axis=None,
                access_pattern="coordinate",
                token_axis=None,
                head_partition=None,
                shared_weight=None,
                update_frequency="odd_step",
                storage_transposed=False,
                structure_class="coordinate",
            )
        )

    return tuple(descriptors)
