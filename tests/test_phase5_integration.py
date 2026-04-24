"""
Phase 5 — Integration tests for auto-stride × warm-start × new retrain policy.

These tests exercise the full pipeline on real (small) trajectory data.
They are marked ``@pytest.mark.integration`` and skipped by default unless
trajectory data is found at the expected location (or one provided via the
``PYGVAMP_TEST_AB42_TRAJ_DIR`` / ``PYGVAMP_TEST_AB42_TOPOLOGY`` env vars).

Run manually with:
    pytest tests/test_phase5_integration.py -v -m integration

The four tests below map to IMPLEMENTATION_PLAN.md Phase 5's list:
  1. Full multi-lag pipeline with all features on.
  2. Each feature can be individually disabled (three sub-configurations).
  3. Reversible model with all features on.
  4. Backward-compatibility: all-off config reproduces pre-Phase-2 behaviour.

Each test uses minimal epochs/stride and a single trajectory file so the
tests complete in a few minutes per run rather than hours.
"""

import json
import os
from pathlib import Path

import pytest

# Reuse the helper from the existing integration-test module to avoid
# duplicating the 100+ lines of encoder-specific config wiring.
from tests.test_pipeline_integration import _make_config


# ---------------------------------------------------------------------------
# Test-data paths — configurable, with sensible defaults for the dev machine.
# ---------------------------------------------------------------------------

AB42_TRAJ_DIR = os.environ.get(
    "PYGVAMP_TEST_AB42_TRAJ_DIR",
    "/mnt/hdd/data/ab42/trajectories/red/r1",
)
AB42_TOPOLOGY = os.environ.get(
    "PYGVAMP_TEST_AB42_TOPOLOGY",
    "/mnt/hdd/data/ab42/trajectories/red/topol.pdb",
)

DATA_AVAILABLE = (
    os.path.isdir(AB42_TRAJ_DIR)
    and os.path.isfile(AB42_TOPOLOGY)
)


# ---------------------------------------------------------------------------
# Shared tiny-config factory for these integration tests.
# ---------------------------------------------------------------------------

def _make_phase5_config(
    tmp_path: Path,
    *,
    lag_times,
    auto_stride: bool,
    warm_start: bool,
    convergence_check: bool = True,
    max_retrains: int = 3,
    reversible: bool = False,
    n_states: int = 4,
    epochs: int = 1,
    stride: int = 1,
    timestep: float = 0.25,  # ab42_red raw timestep in ns
    file_pattern: str = "traj000[0-2].xtc",
):
    """Build a minimal SchNet config that exercises the three new features."""
    cfg = _make_config(
        tmp_path,
        traj_dir=AB42_TRAJ_DIR,
        top=AB42_TOPOLOGY,
        file_pattern=file_pattern,
        selection="name CA",
        stride=stride,
        lag_time=lag_times[0],
        encoder_type="schnet",
        n_states=n_states,
        epochs=epochs,
        timestep=timestep,
    )
    # Multi-lag — plain _make_config only sets the first lag.
    cfg.lag_times = list(lag_times)
    cfg.lag_time = lag_times[0]

    # Phase 2/3/4 flags
    cfg.auto_stride = auto_stride
    cfg.warm_start_retrains = warm_start
    cfg.convergence_check = convergence_check
    cfg.max_retrains = max_retrains
    cfg.reversible = reversible

    # Keep runs fast and cheap.
    cfg.batch_size = 4
    cfg.cpu = True
    cfg.cache = False
    cfg.discover_states = False
    # classifier BN off is already the default in _make_config

    return cfg


def _experiment_outputs_exist(experiment_dir):
    """Minimal smoke check: prep stats, at least one trained model, analysis."""
    exp = Path(experiment_dir)
    prep = exp / "preparation"
    training = exp / "training"
    analysis = exp / "analysis"
    assert prep.exists() and any(prep.iterdir()), f"empty preparation dir: {prep}"
    assert training.exists() and any(training.iterdir()), f"empty training dir: {training}"
    assert analysis.exists() and any(analysis.iterdir()), f"empty analysis dir: {analysis}"
    # Pipeline summary is written at the end of run_complete_pipeline
    assert (exp / "pipeline_summary.json").is_file()


def _prep_frame_dt(experiment_dir) -> float:
    """Return frame_dt_ps persisted by preparation, for auto-stride validation."""
    prep = Path(experiment_dir) / "preparation"
    run_dirs = [d for d in prep.iterdir() if d.is_dir()]
    assert run_dirs, "no preparation run dir"
    stats_path = sorted(run_dirs)[-1] / "dataset_stats.json"
    with open(stats_path) as f:
        stats = json.load(f)
    return stats.get("parameters", {}).get("frame_dt_ps")


# ===========================================================================
# 1. Full multi-lag pipeline, all features on.
# ===========================================================================

@pytest.mark.integration
@pytest.mark.skipif(not DATA_AVAILABLE, reason="ab42 trajectory data not found")
def test_full_multi_lag_pipeline_with_all_features(tmp_path, capsys):
    """All three features on over three lag times.  Verify: pipeline completes,
    prep persisted frame_dt_ps, distinct auto-stride values logged for the
    three lag times, and outputs exist for each."""
    from pygv.pipe.master_pipeline import PipelineOrchestrator

    lag_times = [1.0, 5.0, 10.0]
    cfg = _make_phase5_config(
        tmp_path,
        lag_times=lag_times,
        auto_stride=True,
        warm_start=True,
    )

    orch = PipelineOrchestrator(cfg)
    orch.run_complete_pipeline()

    _experiment_outputs_exist(orch.experiment_dir)

    # frame_dt_ps must be persisted
    frame_dt_ps = _prep_frame_dt(orch.experiment_dir)
    assert frame_dt_ps is not None, "frame_dt_ps not written to dataset_stats.json"
    assert frame_dt_ps == pytest.approx(250.0, rel=0.05), (
        f"expected ab42 timestep ~250 ps, got {frame_dt_ps}"
    )

    # Check that auto-stride produced distinct values per lag time.  The
    # orchestrator logs one "Auto-stride: τ=..." line per (lag × retrain round).
    captured = capsys.readouterr().out
    # Expected strides for frame_dt=250ps, prep_stride=1:
    #   τ=1  -> runtime=1    τ=5  -> runtime=2    τ=10 -> runtime=4
    for tau, expected_rs in [(1.0, 1), (5.0, 2), (10.0, 4)]:
        needle = f"τ={tau:g}ns"
        assert needle in captured, f"auto-stride log missing for {needle}"
        # At least one line for this tau must have stride=expected
        lines = [ln for ln in captured.splitlines()
                 if "Auto-stride" in ln and needle in ln]
        assert any(f"stride={expected_rs}" in ln for ln in lines), (
            f"expected stride={expected_rs} in auto-stride log lines for {needle}, "
            f"got: {lines}"
        )


# ===========================================================================
# 2. Each feature individually disabled.
# ===========================================================================

@pytest.mark.integration
@pytest.mark.skipif(not DATA_AVAILABLE, reason="ab42 trajectory data not found")
@pytest.mark.parametrize("auto_stride,warm_start,label", [
    (True,  False, "auto_stride_only"),
    (False, True,  "warm_start_only"),
    (False, False, "both_off"),
])
def test_features_can_be_individually_disabled(
    tmp_path, auto_stride, warm_start, label
):
    """Three sub-configurations — each must complete without crashing."""
    from pygv.pipe.master_pipeline import PipelineOrchestrator

    cfg = _make_phase5_config(
        tmp_path,
        lag_times=[1.0, 5.0],
        auto_stride=auto_stride,
        warm_start=warm_start,
        max_retrains=2,  # keep per-test wall time short
    )
    orch = PipelineOrchestrator(cfg)
    orch.run_complete_pipeline()
    _experiment_outputs_exist(orch.experiment_dir)


# ===========================================================================
# 3. Reversible model with all features on.
# ===========================================================================

@pytest.mark.integration
@pytest.mark.skipif(not DATA_AVAILABLE, reason="ab42 trajectory data not found")
def test_reversible_model_with_all_features(tmp_path):
    """Warm-starting must correctly rebuild the reversible score module
    when retrains fire.  Smoke test: pipeline completes with --reversible."""
    from pygv.pipe.master_pipeline import PipelineOrchestrator

    cfg = _make_phase5_config(
        tmp_path,
        lag_times=[1.0, 5.0],
        auto_stride=True,
        warm_start=True,
        reversible=True,
        max_retrains=2,
    )
    orch = PipelineOrchestrator(cfg)
    orch.run_complete_pipeline()
    _experiment_outputs_exist(orch.experiment_dir)


# ===========================================================================
# 4. Backward-compatibility — all-off config still works.
# ===========================================================================

@pytest.mark.integration
@pytest.mark.skipif(not DATA_AVAILABLE, reason="ab42 trajectory data not found")
def test_backward_compatibility(tmp_path):
    """A single-lag pipeline with all new features off must still complete,
    matching the pre-Phase-2 invocation shape (no --auto_stride, no warm-start,
    old retrain policy)."""
    from pygv.pipe.master_pipeline import PipelineOrchestrator

    cfg = _make_phase5_config(
        tmp_path,
        lag_times=[2.5],       # single-lag, like pre-Phase-2 runs
        auto_stride=False,
        warm_start=False,
        convergence_check=False,
        max_retrains=2,
    )
    orch = PipelineOrchestrator(cfg)
    orch.run_complete_pipeline()
    _experiment_outputs_exist(orch.experiment_dir)
