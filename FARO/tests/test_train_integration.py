import ast
import copy
import hashlib
import types
from pathlib import Path


REPO_ROOT = Path(__file__).parents[2]
TRAIN_SOURCE = (REPO_ROOT / "train_gpt.py").read_text()
NORMUON_SHA256 = "f61a857d53eb41cec91bfa2621cc97010f6542175c5d8c177d3be16c35be8f4d"


def _class_source(name: str) -> str:
    tree = ast.parse(TRAIN_SOURCE)
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.ClassDef) and item.name == name
    )
    return "\n".join(TRAIN_SOURCE.splitlines()[node.lineno - 1 : node.end_lineno]) + "\n"


def _load_training_symbol(name: str, namespace: dict[str, object]):
    tree = ast.parse(TRAIN_SOURCE)
    node = next(
        (
            item
            for item in tree.body
            if isinstance(item, ast.FunctionDef) and item.name == name
        ),
        None,
    )
    if node is None:
        training_manager = next(
            item
            for item in tree.body
            if isinstance(item, ast.ClassDef) and item.name == "TrainingManager"
        )
        node = next(
            (
                item
                for item in training_manager.body
                if isinstance(item, ast.FunctionDef) and item.name == name
            ),
            None,
        )
    assert node is not None, f"missing training symbol: {name}"
    module = ast.Module(body=[copy.deepcopy(node)], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(REPO_ROOT / "train_gpt.py"), "exec"), namespace)
    return namespace[name]


def test_existing_normuon_source_is_unchanged() -> None:
    digest = hashlib.sha256(_class_source("NorMuonAndAdam").encode()).hexdigest()

    assert digest == NORMUON_SHA256


def test_training_entry_imports_faro_modules() -> None:
    for module in (
        "FARO.experiment",
        "FARO.descriptors",
        "FARO.optimizer",
        "FARO.router",
        "FARO.metrics",
    ):
        assert f"from {module} import" in TRAIN_SOURCE


def test_seed_and_descriptors_are_built_before_model_compile() -> None:
    seed_position = TRAIN_SOURCE.index("seed_everything(experiment_config.seed)")
    model_position = TRAIN_SOURCE.index("model: nn.Module = GPT(")
    descriptor_position = TRAIN_SOURCE.index("build_gpt_descriptors(model)")
    compile_position = TRAIN_SOURCE.index("model: nn.Module = torch.compile(")

    assert seed_position < model_position < descriptor_position < compile_position
    assert TRAIN_SOURCE.count("model: nn.Module = GPT(") == 1


def test_training_manager_has_one_four_way_optimizer_selection() -> None:
    manager = _class_source("TrainingManager")

    assert 'if experiment_config.variant == "normuon"' in manager
    assert 'elif experiment_config.variant in ("adam", "muon")' in manager
    assert "FixedGeometryOptimizer(" in manager
    assert "FAROOptimizer(" in manager
    assert "FARORouter(RouterConfig.from_experiment(experiment_config))" in manager


def test_faro_family_predicate_covers_dynamic_and_fixed_variants() -> None:
    assert 'return experiment_config.variant in {"faro", "faro_fixed"}' in TRAIN_SOURCE
    assert "if _uses_faro_optimizer():\n    configure_faro_observer" in TRAIN_SOURCE
    manager = _class_source("TrainingManager")
    assert manager.count("_uses_faro_optimizer()") >= 4


def test_router_source_has_no_parameter_name_mode_selection() -> None:
    router_source = (REPO_ROOT / "FARO" / "router.py").read_text()

    for forbidden in ('"mlp" in descriptor.label', '"embed" in descriptor.label'):
        assert forbidden not in router_source


def test_disabled_faro_observer_has_layer_index_placeholders() -> None:
    gpt = _class_source("GPT")

    assert "self._faro_attn_indices = (None,) * self._num_attn_layers" in gpt
    assert "self._faro_mlp_indices = (None,) * self.num_layers" in gpt


def test_mlp_fp8_forward_uses_independent_partial_amax_scratch() -> None:
    gpt = _class_source("GPT")

    assert "_mlp_down_partial_amax_scratch_names" in gpt
    assert 'f\"_mlp_down_partial_amax_scratch_{i}\"' in gpt
    assert "partial_amax = getattr(" in gpt
    assert "self._mlp_down_partial_amax[i]" not in gpt


def test_probe_gate_is_shared_by_warmup_and_training_and_stops_after_freeze() -> None:
    manager = _class_source("TrainingManager")

    assert "def should_probe(self, step: int) -> bool:" in manager
    assert "not self.optimizer.all_routes_frozen" in manager
    assert TRAIN_SOURCE.count("training_manager.should_probe(step)") == 2
    assert TRAIN_SOURCE.count("step % experiment_config.probe_interval == 1") == 1


def test_faro_freeze_events_are_finalized_then_drained_after_optimizer_step() -> None:
    manager = _class_source("TrainingManager")

    assert "self.optimizer.finalize_freeze_events()" in manager
    assert "freeze_events = self.optimizer.consume_freeze_events()" in manager
    assert "kind=\"faro_freeze\"" in TRAIN_SOURCE
    assert "event=\"event\"" in TRAIN_SOURCE


def test_freezing_step_finalizes_drains_and_skips_post_update_probe() -> None:
    namespace: dict[str, object] = {
        "_uses_faro_optimizer": lambda: True,
        "experiment_config": types.SimpleNamespace(variant="faro", probe_interval=4),
        "training_schedule": types.SimpleNamespace(get_lr=lambda step: 1.0),
        "get_muon_momentum": lambda step: 0.95,
    }
    step_optimizers = _load_training_symbol("step_optimizers", namespace)
    should_probe = _load_training_symbol("should_probe", namespace)
    should_replay_probe = _load_training_symbol("_should_replay_probe", namespace)
    write_faro_freeze_events = _load_training_symbol(
        "_write_faro_freeze_events", namespace
    )

    lifecycle: list[str] = []

    class FakeFAROOptimizer:
        all_routes_frozen = False

        def step(self, **kwargs):
            lifecycle.append("step")
            self.all_routes_frozen = True
            self.pending = [
                {
                    "step": 5,
                    "newly_frozen": 2,
                    "total_frozen": 2,
                    "mode_counts": {"spectral": 2},
                    "all_routes_frozen": True,
                    "current_allocated_mib": None,
                    "current_reserved_mib": None,
                }
            ]
            return object()

        def finalize_freeze_events(self):
            lifecycle.append("finalize")
            for event in self.pending:
                event["current_allocated_mib"] = 12.5
                event["current_reserved_mib"] = 24.0

        def consume_freeze_events(self):
            lifecycle.append("consume")
            assert all(event["current_allocated_mib"] is not None for event in self.pending)
            events, self.pending = self.pending, []
            return events

    class FakeManager:
        def __init__(self) -> None:
            self.optimizer = FakeFAROOptimizer()
            self.split_step = -1
            self.last_optimizer_result = None

        def _is_adam_step(self, step: int) -> bool:
            return True

    class FakeWriter:
        def __init__(self) -> None:
            self.records: list[dict[str, object]] = []

        def write_events(self, records):
            self.records.extend(records)

    manager = FakeManager()
    manager.should_probe = types.MethodType(should_probe, manager)
    requested_probe = manager.should_probe(5)

    freeze_events = step_optimizers(manager, 5)

    assert lifecycle == ["step", "finalize", "consume"]
    assert requested_probe is True
    assert not should_replay_probe(manager, 5, requested_probe)
    assert manager.optimizer.pending == []

    writer = FakeWriter()
    write_faro_freeze_events(writer, freeze_events)

    assert writer.records == [
        {
            "event": "event",
            "kind": "faro_freeze",
            "message": "FARO routes frozen",
            "step": 5,
            "newly_frozen": 2,
            "total_frozen": 2,
            "mode_counts": {"spectral": 2},
            "all_routes_frozen": True,
            "current_allocated_mib": 12.5,
            "current_reserved_mib": 24.0,
        }
    ]
