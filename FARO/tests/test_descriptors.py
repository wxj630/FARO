from dataclasses import replace

import torch
from torch import nn

from FARO.descriptors import (
    build_gpt_descriptors,
    legal_geometries,
    select_block,
    structure_prior,
)


def _parameter(shape: tuple[int, ...], label: str) -> nn.Parameter:
    parameter = nn.Parameter(torch.randn(shape))
    parameter.label = label
    return parameter


class TinyGPT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_layers = 2
        self._num_attn_layers = 1
        self._num_qk_groups = 6
        self._num_vo_real = 4
        self.qk_bank = _parameter((8, 4, 6), "qk_bank")
        self.qk_bank.reshape = (8, 4, 6)
        self.vo_bank = _parameter((4, 6, 6), "vo_bank")
        self.vo_bank.reshape = (4, 6, 6)
        self.mlp_bank = _parameter((3, 2, 12, 6), "mlp_bank")
        self.mlp_bank.reshape = (6, 12, 6)
        self.embed = nn.Embedding(7, 6)
        self.embed.weight.label = "embed"
        self.lm_head = nn.Linear(7, 6, bias=False)
        self.lm_head.weight.label = "lm_head"
        self.bigram_embed = nn.Embedding(9, 6)
        self.bigram_embed.weight.label = "bigram_embed"
        self.value_embeds = _parameter((14, 6), "value_embeds")
        self.scalars = _parameter((5,), "scalars")
        self.bias = _parameter((), "bias")


def _by_label(model: TinyGPT):
    return {
        descriptor.label: descriptor
        for descriptor in build_gpt_descriptors(model)
    }


def test_registry_splits_real_functional_matrices_and_excludes_padding() -> None:
    by_label = _by_label(TinyGPT())

    assert by_label["mlp_bank.layer0.up"].access_pattern == "dense"
    assert by_label["mlp_bank.layer0.up"].matrix_shape == (12, 6)
    assert by_label["mlp_bank.layer0.down"].input_axis == 0
    assert by_label["mlp_bank.layer0.down"].storage_transposed
    assert len([label for label in by_label if label.startswith("qk_bank.")]) == 6
    assert len([label for label in by_label if label.startswith("vo_bank.")]) == 4
    assert len([label for label in by_label if label.startswith("mlp_bank.")]) == 4
    assert not any("pad" in label for label in by_label)


def test_registry_describes_token_axes_and_shared_weights() -> None:
    by_label = _by_label(TinyGPT())

    assert by_label["embed"].token_axis == 0
    assert by_label["embed"].access_pattern == "token_lookup"
    assert by_label["embed"].shared_weight == "lm_head"
    assert by_label["lm_head"].token_axis == 1
    assert by_label["lm_head"].access_pattern == "token_output"
    assert by_label["lm_head"].shared_weight == "embed"
    assert by_label["bigram_embed"].update_frequency == "odd_step"


def test_legal_geometries_follow_structure_not_label() -> None:
    by_label = _by_label(TinyGPT())
    scalar = by_label["scalars"]
    dense = by_label["mlp_bank.layer0.up"]
    token = by_label["embed"]

    assert legal_geometries(scalar) == ("coordinate",)
    assert legal_geometries(dense) == (
        "coordinate",
        "row_column",
        "spectral",
    )
    assert legal_geometries(token) == (
        "coordinate",
        "row_column",
        "spectral",
    )

    renamed_dense = replace(dense, label="unrelated-A", parameter_label="p0")
    renamed_token = replace(token, label="unrelated-B", parameter_label="p1")
    assert legal_geometries(renamed_dense) == legal_geometries(dense)
    assert legal_geometries(renamed_token) == legal_geometries(token)
    for mode in legal_geometries(dense):
        assert structure_prior(renamed_dense, mode) == structure_prior(dense, mode)
    for mode in legal_geometries(token):
        assert structure_prior(renamed_token, mode) == structure_prior(token, mode)


def test_select_block_returns_live_bank_view() -> None:
    model = TinyGPT()
    descriptor = _by_label(model)["mlp_bank.layer1.down"]

    selected = select_block(model.mlp_bank, descriptor)

    assert selected.shape == (12, 6)
    selected.data.zero_()
    assert torch.count_nonzero(model.mlp_bank[1, 1]) == 0


def test_every_parameter_label_is_covered() -> None:
    model = TinyGPT()
    descriptors = build_gpt_descriptors(model)
    covered = {descriptor.parameter_label for descriptor in descriptors}
    parameter_labels = {parameter.label for parameter in model.parameters()}

    assert covered == parameter_labels
