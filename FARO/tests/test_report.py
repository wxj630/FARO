import json
from pathlib import Path

from FARO.plot_results import FIGURE_NAMES
from FARO.render_report import render_report


FINAL_LOSSES = {
    "adam": 3.5082759857177734,
    "muon": 3.5603034496307373,
    "normuon": 3.2838094234466553,
    "faro": 3.3045642375946045,
    "faro_prior": 3.2774932384490967,
}


def _summary() -> dict[str, object]:
    return {
        "audit": {
            "complete_runs": 15,
            "fairness_hashes_match": True,
            "fairness_hashes": {
                str(seed): {
                    "initialization_hash": f"init-{seed}",
                    "bigram_sign_table_hash": f"bigram-{seed}",
                    "data_manifest_hash": "data-hash",
                }
                for seed in (0, 42, 1024)
            },
            "code_hashes_match": False,
            "code_hashes": {
                "adam": {"0": "code-a0", "42": "code-a1", "1024": "code-a1"},
                "muon": {"0": "code-a0", "42": "code-m1", "1024": "code-m1"},
                "normuon": {
                    "0": "code-a0",
                    "42": "code-m1",
                    "1024": "code-m1",
                },
                "faro": {"0": "code-n0", "42": "code-n1", "1024": "code-n1"},
                "faro_prior": {
                    "0": "code-p0",
                    "42": "code-p1",
                    "1024": "code-p1",
                },
            },
            "settings": {
                "seeds": (0, 42, 1024),
                "world_size": 1,
                "grad_accum_steps": 8,
                "max_steps": 1285,
                "matrix_lr": 0.0015,
                "muon_lr": 0.023,
            },
            "mlp_routing_events": 100,
            "mlp_spectral_ratio": 0.7,
            "prior_mlp_routing_events": 100,
            "prior_mlp_spectral_ratio": 1.0,
        },
        "variants": {
            variant: {
                "final_validation_loss_mean": loss,
                "final_validation_loss_std": 0.0,
                "seed_values": {"0": loss, "42": loss, "1024": loss},
            }
            for variant, loss in FINAL_LOSSES.items()
        },
        "routes": {
            "attention": {"spectral": 60, "coordinate": 30, "row_column": 5},
            "mlp": {"spectral": 61, "coordinate": 115, "row_column": 3},
            "embedding": {"coordinate": 62},
            "lm_head": {"coordinate": 31},
        },
        "route_audit": {
            "event_count": 992,
            "mode_counts": {
                "coordinate": 644,
                "spectral": 323,
                "row_column": 25,
            },
            "switch_count": 116,
            "budget_min": 0.7806143332234627,
            "budget_mean": 0.9601364010920029,
            "budget_max": 1.0,
        },
        "probe_audit": {
            "event_count": 9,
            "predicted_return_mean": 2.150484,
            "realized_return_mean": 2.170747,
            "realized_to_predicted_median": 0.7272254969265689,
        },
        "freeze_audit": {
            "event_count": 11,
            "freeze_steps": [48, 49, 56, 64, 72, 80, 88, 96, 112, 120, 144],
            "all_frozen_step": 144,
            "total_frozen": 118,
            "mode_counts": {"coordinate": 19, "spectral": 98, "row_column": 1},
            "post_freeze_route_events": 0,
            "post_freeze_probe_events": 0,
            "current_allocated_mib": 6831.01513671875,
            "current_reserved_mib": 7202.0,
            "pre_freeze_tokens_per_second_median": 418142.76484316273,
            "post_freeze_tokens_per_second_median": 588031.1973486234,
        },
        "system": {
            variant: {
                "tokens_per_second_mean": 1000.0,
                "training_time_ms_mean": 500000.0,
                "wall_time_ms_mean": 510000.0,
                "peak_allocated_mib_mean": 40000.0,
                "peak_reserved_mib_mean": 45000.0,
            }
            for variant in FINAL_LOSSES
        },
        "source_runs": [
            "/runs/adam-seed0",
            "/runs/muon-seed0",
            "/runs/normuon-seed0",
            "/runs/faro-coordinate-start-fixed-seed0",
            "/runs/faro-sparse-seed0",
        ],
    }


def _with_real_paired_seed_values(summary: dict[str, object]) -> dict[str, object]:
    summary["variants"]["normuon"]["seed_values"] = {
        "0": 3.2838094234466553,
        "42": 3.2773921489715576,
        "1024": 3.2768940925598145,
    }
    summary["variants"]["faro"]["seed_values"] = {
        "0": 3.3045642375946045,
        "42": 3.3191394805908203,
        "1024": 3.339629888534546,
    }
    summary["variants"]["faro_prior"]["seed_values"] = {
        "0": 3.2774932384490967,
        "42": 3.277399778366089,
        "1024": 3.279137372970581,
    }
    return summary


def test_report_contains_five_three_seed_results_and_honest_theory_boundary(
    tmp_path: Path,
) -> None:
    summary_path = tmp_path / "summary.json"
    output = tmp_path / "04-faro-vs-adam-muon.md"
    summary = _with_real_paired_seed_values(_summary())
    summary_path.write_text(json.dumps(summary))

    render_report(summary_path, output)
    report = output.read_text()

    assert "https://kellerjordan.github.io/posts/muon/" in report
    for variant in (
        "Adam",
        "Muon",
        "NorMuonAndAdam",
        "FARO no-prior auto-switch",
        "FARO prior control",
    ):
        assert variant in report
    for value in (
        "3.508275986",
        "3.560303450",
        "3.283809423",
        "3.304564238",
        "3.277493238",
    ):
        assert value in report
    assert "三个 seed（0, 42, 1024）" in report
    assert "同一 seed 内" in report
    assert "seed 0" in report
    assert "init-0" in report
    assert "seed 42" in report
    assert "init-42" in report
    assert "seed 1024" in report
    assert "init-1024" in report
    assert "不能宣称统计显著" in report
    assert "完整代码快照不完全一致" in report
    assert "不是同一完整代码快照下的严格 A/B" in report
    assert "_muon_config" in report
    for difference in ("-0.006316185", "+0.000007629", "+0.002243280"):
        assert difference in report
    assert "不是完整理论 FARO" in report
    assert "跨参数块的联合预算向量" in report
    assert "独立 probe" in report
    assert "embedding / LM-head" in report
    assert "row/column" in report
    assert "992" in report
    assert "116" in report
    assert "0.780614–1.000000" in report
    assert "0.727225" in report
    assert "step 48、49、56" in report
    assert "118" in report
    assert "0 route / 0 probe" in report
    assert "418,142.8" in report
    assert "588,031.2" in report
    for figure in FIGURE_NAMES:
        assert f"results/figures/{figure}" in report
    for index in range(1, 10):
        assert f"### 图 {index}：" in report
    assert "共同的阶段切换" in report
    assert "图 4–7 的路由" in report
    assert "空图" not in report
    # Honest boundary for the no-prior main experiment.
    assert "去人工先验后" in report
    assert "尚未完全去人工化" in report
    assert "不支持“完全不需要人为定义”" in report
    assert "summary.json" in report
    assert all(source in report for source in _summary()["source_runs"])
    assert "FARO-fixed" not in report
    assert "faro_fixed" not in report
    assert "faro-fixed-seed0" not in report
    assert "FARO NorMuon-anchor dynamic" not in report


def test_historical_findings_keep_fixed_ablation_outside_final_comparison() -> None:
    findings = (Path(__file__).parents[1] / "04-experimental-findings.md").read_text()

    assert "FARO-fixed seed0 正式结果" in findings
    assert "FARO-fixed 不再属于最终正式对比" in findings
    assert "FARO explore→freeze dynamic seed0 正式结果" in findings
    assert "3.277493238" in findings
    assert "完整代码快照不一致" in findings
    assert "不是同一完整代码快照下的严格 A/B" in findings
