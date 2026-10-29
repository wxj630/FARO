import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from FARO.experiment import select_adam_pilot


def _pilot(
    root: Path,
    lr: float,
    loss: float,
    *,
    stable: bool = True,
    created_at: datetime,
) -> Path:
    directory = root / str(lr)
    directory.mkdir(parents=True)
    (directory / "summary.json").write_text(
        json.dumps(
            {
                "status": "complete" if stable else "failed",
                "variant": "adam",
                "seed": 0,
                "step": 250,
                "matrix_lr": lr,
                "final_validation_loss": loss,
                "loss_spike": not stable,
                "created_at": created_at.isoformat(),
            }
        )
    )
    return directory


def test_selection_ignores_unstable_and_chooses_lowest_loss(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    directories = [
        _pilot(tmp_path, 0.00075, 3.8, created_at=now),
        _pilot(tmp_path, 0.0015, 3.6, created_at=now),
        _pilot(tmp_path, 0.003, 3.5, stable=False, created_at=now),
    ]

    selection = select_adam_pilot(directories)

    assert selection["selected_lr"] == 0.0015
    assert len(selection["candidates"]) == 3
    assert selection["rule"] == "lowest stable step-250 validation loss; lower LR breaks ties"


def test_selection_uses_lower_lr_as_exact_tie_breaker(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    directories = [
        _pilot(tmp_path, 0.00075, 3.6, created_at=now),
        _pilot(tmp_path, 0.0015, 3.6, created_at=now),
        _pilot(tmp_path, 0.003, 3.7, created_at=now),
    ]

    selection = select_adam_pilot(directories)

    assert selection["selected_lr"] == 0.00075


def test_selection_rejects_missing_declared_lr(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    directories = [
        _pilot(tmp_path, 0.00075, 3.6, created_at=now),
        _pilot(tmp_path, 0.0015, 3.5, created_at=now),
    ]

    with pytest.raises(ValueError, match="exactly the declared Adam LRs"):
        select_adam_pilot(directories)


def test_selection_refuses_if_faro_started_before_freeze(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    directories = [
        _pilot(tmp_path, 0.00075, 3.6, created_at=now),
        _pilot(tmp_path, 0.0015, 3.5, created_at=now),
        _pilot(tmp_path, 0.003, 3.7, created_at=now),
    ]
    faro = tmp_path / "faro" / "summary.json"
    faro.parent.mkdir()
    faro.write_text(
        json.dumps(
            {
                "variant": "faro",
                "created_at": (now - timedelta(minutes=1)).isoformat(),
            }
        )
    )

    with pytest.raises(ValueError, match="FARO run started before Adam LR was frozen"):
        select_adam_pilot(directories, faro_summaries=[faro])
