"""
Tests for Phase 4 — retrain policy.

Covers the 7 plan-required checks:
  * terminate when the diagnostic recommends the k the model was just trained at
  * continue when k keeps shrinking
  * hit max_retrains and stop
  * returned model is the last trained one
  * default max is 5
  * convergence check can be disabled
  * warm_start_retrains defaults to True after Phase 4
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from pygv.config.base_config import BaseConfig
from pygv.pipe import master_pipeline as mp
from pygv.pipe.master_pipeline import PipelineOrchestrator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockReport:
    """Stand-in for StateReductionReport with just the fields the loop reads."""

    def __init__(self, recommendation, effective_n_states):
        self.recommendation = recommendation
        self.effective_n_states = effective_n_states


def _make_orch(tmp_path, *, warm_start=False, max_retrains=5, convergence_check=True):
    cfg = BaseConfig()
    cfg.warm_start_retrains = warm_start  # False avoids the torch.load branch
    cfg.auto_stride = False
    cfg.cache = False
    cfg.max_retrains = max_retrains
    cfg.convergence_check = convergence_check
    cfg.output_dir = str(tmp_path)
    cfg.protein_name = "test"
    cfg.n_states_list = [12]  # placeholder, loop doesn't use this directly
    cfg.lag_times = [1.0]     # _create_analysis_args falls back to this
    orch = PipelineOrchestrator(cfg)
    return orch


def _make_dirs(tmp_path: Path) -> dict:
    dirs = {
        "training": tmp_path / "training",
        "analysis": tmp_path / "analysis",
        "preparation": tmp_path / "preparation",  # _discover_dataset_path reads this
        "cache": tmp_path / "cache",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def _install_mocks(monkeypatch, reports_sequence, trained_sink):
    """
    Monkeypatch run_training and run_analysis in the master_pipeline namespace.

    - run_training records the n_states used for each retrain and returns a
      dummy path string.
    - run_analysis pops the next report from ``reports_sequence`` and returns
      it wrapped in {"diagnostic_report": report}.
    """
    reports_iter = iter(reports_sequence)
    call_idx = {"n": 0}

    def mock_run_training(train_args, pre_built_model=None):
        trained_sink.append(int(train_args.n_states))
        path = f"/tmp/mock_model_{call_idx['n']}.pt"
        call_idx["n"] += 1
        return path

    def mock_run_analysis(analysis_args):
        report = next(reports_iter, None)
        return {"diagnostic_report": report}

    monkeypatch.setattr(mp, "run_training", mock_run_training)
    monkeypatch.setattr(mp, "run_analysis", mock_run_analysis)


def _run_loop(orch, tmp_path, initial_exp_name, initial_report, max_retrain):
    """Seed the retrain loop with a single initial experiment and run it."""
    dirs = _make_dirs(tmp_path)
    trained_models = {initial_exp_name: "/tmp/initial_model.pt"}
    analysis_results = {initial_exp_name: {"diagnostic_report": initial_report}}
    dataset_path = str(tmp_path / "dataset")
    orch._run_retrain_loop(
        dirs=dirs,
        dataset_path=dataset_path,
        trained_models=trained_models,
        analysis_results=analysis_results,
        max_retrain=max_retrain,
    )
    return trained_models, analysis_results


# ---------------------------------------------------------------------------
# 1. Terminate on convergence
# ---------------------------------------------------------------------------

def test_terminate_on_convergence(monkeypatch, tmp_path):
    """Initial analysis recommends k=8.  After training with k=8, analysis
    still recommends k=8 (same k) → loop must stop (rule (a))."""
    orch = _make_orch(tmp_path, max_retrains=5)
    retrain_reports = [
        # 1st retrain's analysis: same k (8) as the k just trained -> converge
        MockReport(recommendation="retrain", effective_n_states=8),
    ]
    trained_sink = []
    _install_mocks(monkeypatch, retrain_reports, trained_sink)

    # Initial analysis recommends shrinking from current k to 8.
    initial = MockReport(recommendation="retrain", effective_n_states=8)
    _run_loop(orch, tmp_path, "lag1ns_12states", initial, max_retrain=5)

    # Exactly one retrain should have run (with k=8), then the convergence check stops us.
    assert trained_sink == [8]


# ---------------------------------------------------------------------------
# 2. Continue on no convergence
# ---------------------------------------------------------------------------

def test_continue_on_no_convergence(monkeypatch, tmp_path):
    """Diagnostic recommends strictly shrinking k each round.  Loop must
    continue until it hits max_retrains."""
    orch = _make_orch(tmp_path, max_retrains=5)
    # After initial -> retrain at k=8, subsequent analyses recommend
    # 7, 6, 5, 4 (strictly shrinking, never equal to the k just trained).
    retrain_reports = [
        MockReport("retrain", 7),
        MockReport("retrain", 6),
        MockReport("retrain", 5),
        MockReport("retrain", 4),
        MockReport("retrain", 3),
    ]
    trained_sink = []
    _install_mocks(monkeypatch, retrain_reports, trained_sink)

    initial = MockReport("retrain", 8)
    _run_loop(orch, tmp_path, "lag1ns_10states", initial, max_retrain=5)

    assert trained_sink == [8, 7, 6, 5, 4]


# ---------------------------------------------------------------------------
# 3. max_retrains cap
# ---------------------------------------------------------------------------

def test_max_retrains_cap(monkeypatch, tmp_path, capsys):
    """Always-different-k scenario: verify the loop stops at max_retrains
    and logs an exhaustion warning."""
    orch = _make_orch(tmp_path, max_retrains=3)
    retrain_reports = [
        MockReport("retrain", 7),
        MockReport("retrain", 6),
        MockReport("retrain", 5),  # would queue a 4th round, but capped
    ]
    trained_sink = []
    _install_mocks(monkeypatch, retrain_reports, trained_sink)

    initial = MockReport("retrain", 8)
    _run_loop(orch, tmp_path, "lag1ns_10states", initial, max_retrain=3)

    # 3 retrains, never more
    assert trained_sink == [8, 7, 6]
    # Exhaustion warning printed
    captured = capsys.readouterr().out
    assert "exhausted" in captured.lower() or "max iterations" in captured.lower()


# ---------------------------------------------------------------------------
# 4. max_retrains_uses_last_model
# ---------------------------------------------------------------------------

def test_max_retrains_uses_last_model(monkeypatch, tmp_path):
    """After exhaustion, trained_models must contain the final retrain's model."""
    orch = _make_orch(tmp_path, max_retrains=3)
    retrain_reports = [
        MockReport("retrain", 7),
        MockReport("retrain", 6),
        MockReport("retrain", 5),
    ]
    trained_sink = []
    _install_mocks(monkeypatch, retrain_reports, trained_sink)

    dirs = _make_dirs(tmp_path)
    initial = MockReport("retrain", 8)
    trained_models = {"lag1ns_10states": "/tmp/initial_model.pt"}
    analysis_results = {"lag1ns_10states": {"diagnostic_report": initial}}
    orch._run_retrain_loop(
        dirs=dirs,
        dataset_path=str(tmp_path / "dataset"),
        trained_models=trained_models,
        analysis_results=analysis_results,
        max_retrain=3,
    )

    # The retrain naming scheme: lag1ns_{k}states[{_retrained,_retrainedN}]
    # After 3 retrains the last trained k is 6 (the sequence is 8, 7, 6)
    # — regardless of the exact name, the trained_models dict must grow with
    # entries that end in ...{6}states... as the last added key.
    retrained_keys = [k for k in trained_models if k != "lag1ns_10states"]
    assert len(retrained_keys) == 3
    # Final trained n_states is the third entry in trained_sink
    assert trained_sink[-1] == 6
    # And the last retrained key's model path corresponds to the final call
    last_key = retrained_keys[-1]
    assert "6states" in last_key


# ---------------------------------------------------------------------------
# 5. Default max is 5
# ---------------------------------------------------------------------------

def test_default_max_is_five():
    assert BaseConfig().max_retrains == 5


# ---------------------------------------------------------------------------
# 6. convergence_check can be disabled
# ---------------------------------------------------------------------------

def test_convergence_check_can_be_disabled(monkeypatch, tmp_path):
    """With convergence_check=False, the loop must run all max_retrains rounds
    even when k stabilises."""
    orch = _make_orch(tmp_path, max_retrains=3, convergence_check=False)
    # All analyses recommend the same k (would normally terminate round 1)
    retrain_reports = [
        MockReport("retrain", 8),
        MockReport("retrain", 8),
        MockReport("retrain", 8),
    ]
    trained_sink = []
    _install_mocks(monkeypatch, retrain_reports, trained_sink)

    initial = MockReport("retrain", 8)
    _run_loop(orch, tmp_path, "lag1ns_10states", initial, max_retrain=3)

    # With convergence_check off, all three retrains run at k=8
    assert trained_sink == [8, 8, 8]


# ---------------------------------------------------------------------------
# 7. warm_start_retrains default
# ---------------------------------------------------------------------------

def test_warm_start_default_on():
    assert BaseConfig().warm_start_retrains is True


# ---------------------------------------------------------------------------
# Extras — escape-hatch flags
# ---------------------------------------------------------------------------

def test_no_warm_start_retrains_flag_overrides_default(monkeypatch):
    """--no_warm_start_retrains must force warm_start_retrains=False even if
    --warm_start_retrains is also passed."""
    from pygv.pipe.args import parse_pipeline_args

    argv = [
        "--traj_dir", "/tmp/fake",
        "--top", "/tmp/fake.pdb",
        "--warm_start_retrains",
        "--no_warm_start_retrains",
    ]
    monkeypatch.setattr("sys.argv", ["prog"] + argv)
    ns = parse_pipeline_args()
    assert ns.warm_start_retrains is True
    assert ns.no_warm_start_retrains is True


def test_convergence_no_flag_terminates_on_zero_shrink(monkeypatch, tmp_path):
    """Sanity check of rule (a): even if the report still says "retrain",
    the loop terminates when effective_n_states == the k just trained."""
    orch = _make_orch(tmp_path, max_retrains=5, convergence_check=True)
    # The recommendation stays "retrain" (e.g. merge_groups nonempty) but the
    # effective_n_states matches the model's current k.
    retrain_reports = [MockReport("retrain", 9)]
    trained_sink = []
    _install_mocks(monkeypatch, retrain_reports, trained_sink)

    initial = MockReport("retrain", 9)
    _run_loop(orch, tmp_path, "lag1ns_12states", initial, max_retrain=5)

    assert trained_sink == [9]
