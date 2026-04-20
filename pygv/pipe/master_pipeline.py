#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Master Pipeline Orchestrator for PyGVAMP

This script orchestrates the complete VAMPNet pipeline:
1. Data Preparation
2. Training (multiple lag times and n_states)
3. Analysis

Supports configuration presets and selective re-running of pipeline stages.
"""

import os
import re
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

# Import pipeline components
from pygv.pipe.preparation import run_preparation
from pygv.pipe.training import run_training
from pygv.pipe.analysis import run_analysis
from pygv.config import get_config, list_presets
from pygv.pipe.args import parse_pipeline_args
from pygv.utils.logging_utils import PipelineLogger

class PipelineOrchestrator:
    """
    Orchestrates the complete VAMPNet pipeline with caching and selective execution
    """

    def __init__(self, config):
        """
        Initialize pipeline orchestrator

        Parameters
        ----------
        config : BaseConfig
            Configuration object containing all pipeline parameters
        """
        self.config = config
        self.experiment_dir = None

    def setup_experiment_directory(self):
        """Create experiment directory structure"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"exp_{self.config.protein_name}_{timestamp}"

        self.experiment_dir = Path(self.config.output_dir) / exp_name

        # Create subdirectories
        dirs = {
            'root': self.experiment_dir,
            'preparation': self.experiment_dir / 'preparation',
            'training': self.experiment_dir / 'training',
            'analysis': self.experiment_dir / 'analysis',
            'cache': self.experiment_dir / 'cache' if self.config.cache else None,
            'logs': self.experiment_dir / 'logs'
        }

        for name, path in dirs.items():
            if path is not None:
                path.mkdir(parents=True, exist_ok=True)

        # Save configuration
        config_path = self.experiment_dir / 'config.yaml'
        self.config.to_yaml(str(config_path))

        print(f"Experiment directory created: {self.experiment_dir}")
        return dirs

    def run_preparation_phase(self, dirs):
        """
        Run data preparation phase.

        Returns
        -------
        tuple
            (dataset_path, recommended_n_states) where recommended_n_states
            is an int if state discovery ran, or None otherwise.
        """
        print("\n" + "=" * 60)
        print("PHASE 1: DATA PREPARATION")
        print("=" * 60)

        # Calculate optimal stride based on hurry mode
        if self.config.hurry:
            optimal_stride = self._calculate_optimal_stride()
            print(f"Hurry mode enabled: using stride={optimal_stride}")
            self.config.stride = optimal_stride

        # Run preparation
        prep_args = self._create_prep_args(dirs)
        dataset_path = run_preparation(prep_args)

        # Read back recommended n_states from state discovery if it ran
        recommended_n_states = None
        if self.config.discover_states:
            stats_path = os.path.join(dataset_path, 'dataset_stats.json')
            if os.path.isfile(stats_path):
                with open(stats_path) as f:
                    stats = json.load(f)
                if 'state_discovery' in stats:
                    recommended_n_states = stats['state_discovery'].get('recommended_n_states')
                    if recommended_n_states is not None:
                        print(f"State discovery recommended n_states = {recommended_n_states}")

        return dataset_path, recommended_n_states

    def run_training_phase(self, dirs, dataset_path):
        """
        Run training phase for all lag times and n_states combinations

        Returns dictionary of trained models
        """
        print("\n" + "=" * 60)
        print("PHASE 2: TRAINING")
        print("=" * 60)

        trained_models = {}

        # Get lag times and n_states lists
        lag_times = self.config.lag_times if isinstance(self.config.lag_times, list) else [self.config.lag_times]
        n_states_list = self.config.n_states_list if hasattr(self.config, 'n_states_list') else [self.config.n_states]

        total_experiments = len(lag_times) * len(n_states_list)
        print(f"Training {total_experiments} models:")
        print(f"  Lag times: {lag_times}")
        print(f"  N states: {n_states_list}")

        experiment_count = 0
        for lag_time in lag_times:
            for n_states in n_states_list:
                experiment_count += 1
                print(f"\n--- Experiment {experiment_count}/{total_experiments} ---")
                print(f"Lag time: {lag_time} ns, N states: {n_states}")

                # Create experiment-specific directory
                exp_name = f"lag{lag_time}ns_{n_states}states"
                exp_dir = dirs['training'] / exp_name
                exp_dir.mkdir(exist_ok=True)

                # Check if model already exists (in timestamped subdirs or direct)
                existing = sorted(exp_dir.glob('*/models/best_model.pt'))
                if existing:
                    model_path = existing[-1]  # latest run
                    print(f"Model already exists: {model_path}")
                    trained_models[exp_name] = str(model_path)
                    continue
                if (exp_dir / 'best_model.pt').exists():
                    print(f"Model already exists: {exp_dir / 'best_model.pt'}")
                    trained_models[exp_name] = str(exp_dir / 'best_model.pt')
                    continue

                # Create training args
                train_args = self._create_train_args(
                    dirs,
                    dataset_path,
                    lag_time,
                    n_states,
                    exp_dir
                )

                # Run training
                try:
                    model_path = run_training(train_args)
                    trained_models[exp_name] = str(model_path)
                    print(f"Training completed: {model_path}")
                except Exception as e:
                    print(f"Training failed: {str(e)}")
                    continue

        return trained_models

    def run_analysis_phase(self, dirs, trained_models):
        """
        Run analysis phase for all trained models
        """
        print("\n" + "=" * 60)
        print("PHASE 3: ANALYSIS")
        print("=" * 60)

        analysis_results = {}

        for exp_name, model_path in trained_models.items():
            print(f"\nAnalyzing: {exp_name}")

            # Create analysis directory
            analysis_dir = dirs['analysis'] / exp_name
            analysis_dir.mkdir(exist_ok=True)

            # Create analysis args
            analysis_args = self._create_analysis_args(
                dirs,
                model_path,
                analysis_dir
            )

            # Run analysis
            try:
                results = run_analysis(analysis_args)
                analysis_results[exp_name] = results
                print(f"Analysis completed: {analysis_dir}")
            except Exception as e:
                print(f"Analysis failed: {str(e)}")
                continue

        return analysis_results

    def _calculate_optimal_stride(self):
        """
        Calculate optimal stride for hurry mode using actual trajectory timestep

        Tries stride=10 first, then reduces until compatible with lag times
        """
        max_stride = 50
        min_lag_time = min(self.config.lag_times) if isinstance(self.config.lag_times, list) else self.config.lag_times

        # Get timestep (user override or inferred from trajectory)
        if self.config.timestep is not None:
            timestep_ns = self.config.timestep
            print(f"Using user-specified timestep: {timestep_ns} ns")
        else:
            try:
                import mdtraj as md
                traj_iterator = md.iterload(self.config.traj_dir, top=self.config.top, chunk=2)
                first_chunk = next(traj_iterator)

                if len(first_chunk.time) < 2:
                    raise ValueError("Trajectory must have at least 2 frames to infer timestep")

                timestep_ps = first_chunk.time[1] - first_chunk.time[0]
                timestep_ns = timestep_ps / 1000.0

                print(f"Inferred trajectory timestep: {timestep_ps:.3f} ps ({timestep_ns:.6f} ns)")

            except Exception as e:
                print(f"Warning: Could not infer timestep from trajectory: {str(e)}")
                print("Falling back to assumed timestep of 0.1 ns")
                timestep_ns = 0.1

        # Convert min_lag_time to same units (nanoseconds)
        min_lag_time_ns = min_lag_time

        # Try different stride values from max down to 1
        for stride in range(max_stride, 0, -1):
            effective_timestep_ns = timestep_ns * stride

            # Check if lag time is divisible by effective timestep
            # Use small epsilon for floating point comparison
            remainder = min_lag_time_ns % effective_timestep_ns

            if remainder < 1e-6 or abs(remainder - effective_timestep_ns) < 1e-6:
                print(f"Selected stride: {stride} (effective timestep: {effective_timestep_ns:.6f} ns)")
                return stride

        # If no compatible stride found, return 1 as fallback
        print(f"Warning: No compatible stride found. Using stride=1")
        return 1

    def _create_prep_args(self, dirs):
        """Create arguments for preparation phase"""
        args = argparse.Namespace()

        # Data paths
        args.traj_dir = self.config.traj_dir
        args.top = self.config.top
        args.file_pattern = self.config.file_pattern
        args.recursive = self.config.recursive

        # Processing parameters
        args.selection = self.config.selection
        args.stride = self.config.stride
        args.lag_time = min(self.config.lag_times) if isinstance(self.config.lag_times, list) else self.config.lag_time
        args.n_neighbors = self.config.n_neighbors
        args.node_embedding_dim = self.config.node_embedding_dim
        args.gaussian_expansion_dim = self.config.gaussian_expansion_dim

        # Output
        args.output_dir = str(dirs['preparation'])
        args.cache_dir = str(dirs['cache']) if self.config.cache else None
        args.use_cache = self.config.cache

        # State discovery
        args.discover_states = self.config.discover_states
        args.g2v_embedding_dim = self.config.g2v_embedding_dim
        args.g2v_max_degree = self.config.g2v_max_degree
        args.g2v_epochs = self.config.g2v_epochs
        args.g2v_min_count = self.config.g2v_min_count
        args.g2v_umap_dim = self.config.g2v_umap_dim or [2, 3, 5, 6, 7]
        args.min_states = self.config.min_states
        args.max_states = self.config.max_states
        args.batch_size = self.config.batch_size
        args.timestep = self.config.timestep

        return args

    def _create_train_args(self, dirs, dataset_path, lag_time, n_states, exp_dir):
        """Create arguments for training phase"""
        # Start with config as base
        args = argparse.Namespace(**self.config.to_dict())

        # Override specific parameters
        args.lag_time = lag_time
        args.n_states = n_states
        args.output_dir = str(exp_dir)
        args.cache_dir = dataset_path if self.config.cache else None
        args.use_cache = self.config.cache

        return args

    def _create_analysis_args(self, dirs, model_path, analysis_dir):
        """Create arguments for analysis phase"""
        args = argparse.Namespace()

        # Model and data
        args.model_path = model_path
        args.traj_dir = self.config.traj_dir
        args.top = self.config.top
        args.file_pattern = self.config.file_pattern

        # Processing parameters (match training)
        args.selection = self.config.selection
        args.stride = self.config.stride
        args.n_neighbors = self.config.n_neighbors
        args.node_embedding_dim = self.config.node_embedding_dim
        args.gaussian_expansion_dim = self.config.gaussian_expansion_dim

        # Analysis parameters
        args.analysis_dir = str(analysis_dir)
        args.protein_name = self.config.protein_name
        args.batch_size = self.config.batch_size
        args.cpu = self.config.cpu
        args.cache_dir = str(dirs['cache']) if self.config.cache else None

        # Timestep override
        args.timestep = self.config.timestep

        # State diagnostics parameters
        args.population_threshold = self.config.population_threshold
        args.jsd_threshold = self.config.jsd_threshold

        # Visualization topology (full protein, no water/lipids)
        args.viz_topology = None
        dataset_path = self._discover_dataset_path(dirs)
        if dataset_path:
            viz_path = os.path.join(dataset_path, 'visualization_topology.pdb')
            if os.path.isfile(viz_path):
                args.viz_topology = viz_path

        # Get lag_time from model path
        match = re.search(r'lag(\d+)ns', str(model_path))
        if match:
            args.lag_time = float(match.group(1))
        else:
            args.lag_time = self.config.lag_times[0] if isinstance(self.config.lag_times,
                                                                   list) else self.config.lag_time

        return args

    def _discover_trained_models(self, dirs) -> dict:
        """Scan training directory for existing trained models."""
        trained_models = {}
        training_dir = dirs['training']
        if not training_dir.exists():
            return trained_models
        for exp_dir in sorted(training_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            # Find best_model.pt inside timestamped subdirs
            candidates = sorted(exp_dir.glob('*/models/best_model.pt'))
            if candidates:
                # Use the latest (last sorted) timestamped run
                trained_models[exp_dir.name] = str(candidates[-1])
            # Also check direct path (orchestrator's own layout)
            elif (exp_dir / 'best_model.pt').exists():
                trained_models[exp_dir.name] = str(exp_dir / 'best_model.pt')
        return trained_models

    def _discover_dataset_path(self, dirs) -> Optional[str]:
        """Find existing dataset in preparation directory."""
        prep_dir = dirs['preparation']
        if not prep_dir.exists():
            return None
        # Preparation creates timestamped subdirs with dataset files
        candidates = sorted(prep_dir.iterdir())
        for c in reversed(candidates):  # latest first
            if c.is_dir() and (c / 'dataset_stats.json').exists():
                return str(c)
        # Fallback: the prep dir itself if it contains data
        if (prep_dir / 'dataset_stats.json').exists():
            return str(prep_dir)
        return None

    def resume_experiment_directory(self, experiment_dir: str) -> dict:
        """Resume from an existing experiment directory."""
        exp_path = Path(experiment_dir)
        # Resolve relative names against output_dir
        if not exp_path.is_absolute():
            exp_path = Path(self.config.output_dir) / experiment_dir
        self.experiment_dir = exp_path
        dirs = {
            'root': self.experiment_dir,
            'preparation': self.experiment_dir / 'preparation',
            'training': self.experiment_dir / 'training',
            'analysis': self.experiment_dir / 'analysis',
            'cache': self.experiment_dir / 'cache' if self.config.cache else None,
            'logs': self.experiment_dir / 'logs',
        }
        for name, path in dirs.items():
            if path is not None:
                path.mkdir(parents=True, exist_ok=True)
        print(f"Resuming experiment: {self.experiment_dir}")
        return dirs

    def _cleanup_dataset(self, dataset_path):
        """Remove dataset files if not caching"""
        print(f"Cleaning up dataset: {dataset_path}")
        try:
            if os.path.isfile(dataset_path):
                os.remove(dataset_path)
            elif os.path.isdir(dataset_path):
                import shutil
                shutil.rmtree(dataset_path)
        except Exception as e:
            print(f"Warning: Could not clean up dataset: {str(e)}")

    def _run_retrain_loop(self, dirs, dataset_path, trained_models,
                          analysis_results, max_retrain=2):
        """
        Check analysis results for "retrain" recommendations and automatically
        retrain with the reduced state count.

        Mutates trained_models and analysis_results in-place.

        Parameters
        ----------
        dirs : dict
            Experiment directory structure.
        dataset_path : str
            Path to the prepared dataset.
        trained_models : dict
            Maps exp_name -> model_path. Updated with retrained models.
        analysis_results : dict
            Maps exp_name -> results dict. Updated with retrained analysis.
        max_retrain : int
            Maximum number of retrain iterations per experiment (default: 2).
        """
        # Collect experiments that need retraining from the initial analysis
        pending = []
        for exp_name, results in list(analysis_results.items()):
            report = results.get("diagnostic_report")
            if report is not None and getattr(report, "recommendation", None) == "retrain":
                pending.append((exp_name, report.effective_n_states, 0))

        if not pending:
            return

        print("\n" + "=" * 60)
        print("PHASE 3b: AUTOMATIC RETRAINING")
        print("=" * 60)

        while pending:
            exp_name, new_n_states, iteration = pending.pop(0)

            # Extract lag_time from the original exp_name
            match = re.search(r'lag([\d.]+)ns', exp_name)
            if not match:
                print(f"Could not parse lag_time from '{exp_name}', skipping retrain.")
                continue
            lag_time = float(match.group(1))

            # Build retrained exp_name with suffix
            suffix = "_retrained" if iteration == 0 else f"_retrained{iteration + 1}"
            retrained_exp = f"lag{lag_time:g}ns_{new_n_states}states{suffix}"

            print(f"\n--- Retraining: {exp_name} -> {retrained_exp} "
                  f"(n_states {exp_name} -> {new_n_states}, iteration {iteration + 1}/{max_retrain}) ---")

            # Create training directory
            exp_dir = dirs['training'] / retrained_exp
            exp_dir.mkdir(exist_ok=True)

            # Train
            train_args = self._create_train_args(
                dirs, dataset_path, lag_time, new_n_states, exp_dir
            )
            try:
                model_path = run_training(train_args)
                trained_models[retrained_exp] = str(model_path)
                print(f"Retrain training completed: {model_path}")
            except Exception as e:
                print(f"Retrain training failed for {retrained_exp}: {e}")
                continue

            # Analyze
            analysis_dir = dirs['analysis'] / retrained_exp
            analysis_dir.mkdir(exist_ok=True)
            analysis_args = self._create_analysis_args(dirs, model_path, analysis_dir)
            try:
                results = run_analysis(analysis_args)
                analysis_results[retrained_exp] = results
                print(f"Retrain analysis completed: {analysis_dir}")
            except Exception as e:
                print(f"Retrain analysis failed for {retrained_exp}: {e}")
                continue

            # Check if this retrained model also recommends "retrain"
            report = results.get("diagnostic_report")
            if (report is not None
                    and getattr(report, "recommendation", None) == "retrain"
                    and iteration + 1 < max_retrain):
                print(f"Retrained model also recommends retrain "
                      f"(effective_n_states={report.effective_n_states}). "
                      f"Queuing iteration {iteration + 2}.")
                pending.append((retrained_exp, report.effective_n_states, iteration + 1))
            elif (report is not None
                  and getattr(report, "recommendation", None) == "retrain"):
                print(f"Retrained model still recommends retrain but max iterations "
                      f"({max_retrain}) reached. Stopping.")

    def run_complete_pipeline(self, skip_preparation=False, skip_training=False,
                              only_analysis=False, resume=None):
        """Run the complete pipeline with optional phase skipping and resume support."""
        # Setup dirs early so we have the logs directory
        if resume:
            dirs = self.resume_experiment_directory(resume)
        else:
            dirs = self.setup_experiment_directory()

        # Start logging
        self._logger = PipelineLogger(log_dir=str(dirs['logs']))
        self._logger.start()

        try:
            print("=" * 60)
            print("STARTING PYGVAMP PIPELINE")
            print("=" * 60)
            print(f"Protein: {self.config.protein_name}")
            print(f"Model: {self.config.encoder_type}")
            print(f"Preset: {getattr(self.config, 'preset', 'custom')}")
            print(f"Cache: {self.config.cache}")
            print(f"Hurry mode: {self.config.hurry}")

            dataset_path = None
            trained_models = {}

            # Phase 1: Preparation
            if not skip_preparation and not only_analysis:
                dataset_path, recommended_n_states = self.run_preparation_phase(dirs)
                # Use discovered n_states if no explicit list was provided
                if recommended_n_states is not None and not hasattr(self.config, '_n_states_from_cli'):
                    print(f"Using discovered n_states = {recommended_n_states}")
                    self.config.n_states_list = [recommended_n_states]
            else:
                print("Skipping preparation phase.")
                dataset_path = self._discover_dataset_path(dirs)

            # Phase 2: Training
            if not skip_training and not only_analysis:
                if dataset_path is None:
                    print("Error: No dataset found. Run preparation first or provide --cache.")
                    return None
                trained_models = self.run_training_phase(dirs, dataset_path)
            else:
                print("Skipping training phase.")

            # Discover all trained models (merges with any just-trained ones)
            discovered = self._discover_trained_models(dirs)
            trained_models = {**discovered, **trained_models}  # just-trained wins on conflict

            # Phase 3: Analysis
            if trained_models:
                analysis_results = self.run_analysis_phase(dirs, trained_models)
            else:
                print("No trained models found. Run training first.")
                analysis_results = {}

            # Phase 3b: Automatic retraining (skip when only running analysis)
            if analysis_results and dataset_path and not only_analysis:
                self._run_retrain_loop(dirs, dataset_path, trained_models, analysis_results)

            # Save summary
            self._save_pipeline_summary(dirs, trained_models, analysis_results)

            print("\n" + "=" * 60)
            print("PIPELINE COMPLETED")
            print("=" * 60)
            print(f"Results saved to: {self.experiment_dir}")
            print(f"Log file: {self._logger.log_path}")

            return {
                'experiment_dir': self.experiment_dir,
                'trained_models': trained_models,
                'analysis_results': analysis_results
            }
        finally:
            self._logger.stop()

    def _save_pipeline_summary(self, dirs, trained_models, analysis_results):
        """Save pipeline execution summary"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config.to_dict(),
            'trained_models': trained_models,
            'analysis_completed': list(analysis_results.keys())
        }

        summary_path = self.experiment_dir / 'pipeline_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)


SUPPORTED_TOPOLOGY_EXTENSIONS = {
    '.pdb', '.pdb.gz', '.gro',
    '.pdbx', '.pdbx.gz', '.mmcif', '.mmcif.gz', '.cif', '.cif.gz',
}


def validate_topology_file(top_path: str):
    """
    Validate that the topology file has coordinates (required for standalone loading).

    Topology-only formats like .psf (CHARMM/NAMD), .prmtop (AMBER) are not supported
    because several pipeline steps need to load the file with md.load() which requires
    coordinates.

    Raises
    ------
    SystemExit
        If the topology file extension is not supported.
    """
    import sys
    ext = ''.join(Path(top_path).suffixes).lower()  # handles .pdb.gz
    if ext not in SUPPORTED_TOPOLOGY_EXTENSIONS:
        sys.exit(
            f"Error: Topology file '{top_path}' has unsupported extension '{ext}'.\n"
            f"The pipeline requires a topology file with coordinates.\n"
            f"Supported formats: {', '.join(sorted(SUPPORTED_TOPOLOGY_EXTENSIONS))}\n"
            f"Topology-only formats (.psf, .prmtop) are not supported.\n"
            f"Hint: Convert your topology to PDB, e.g.:\n"
            f"  mdtraj: md.load('traj.dcd', top='file.psf')[0].save_pdb('topology.pdb')\n"
            f"  VMD:    mol load psf file.psf; animate write pdb topology.pdb"
        )


def _warn_if_timestep_unreasonable(timestep_ns: float, n_traj_files: int):
    """
    Print a warning if the inferred timestep looks suspiciously large or small.

    Some MD engines (notably NAMD) write XTC/DCD time metadata in non-standard
    units (e.g. femtoseconds instead of picoseconds), causing a 1000x error.
    """
    if timestep_ns >= 10.0:
        print(
            f"\n{'=' * 60}\n"
            f"WARNING: Inferred timestep is {timestep_ns:.1f} ns per frame.\n"
            f"This is unusually large and may indicate incorrect time metadata\n"
            f"in the trajectory file (common with NAMD/CHARMM trajectories).\n"
            f"\n"
            f"If this is wrong, re-run with --timestep <correct_value_in_ns>\n"
            f"For example: --timestep 0.2  (for 0.2 ns = 200 ps between frames)\n"
            f"{'=' * 60}\n"
        )
    elif timestep_ns <= 1e-6:
        print(
            f"\n{'=' * 60}\n"
            f"WARNING: Inferred timestep is {timestep_ns:.2e} ns per frame.\n"
            f"This is unusually small and may indicate incorrect time metadata.\n"
            f"\n"
            f"If this is wrong, re-run with --timestep <correct_value_in_ns>\n"
            f"{'=' * 60}\n"
        )


def validate_lag_times(config):
    """
    Validate that all requested lag times are compatible with the trajectory timestep.

    Loads 2 frames from the first trajectory file to infer the timestep, then
    checks that each lag time is an exact multiple of (timestep * stride).

    Raises
    ------
    SystemExit
        If any lag time is incompatible.
    """
    import sys
    import mdtraj as md
    from pygv.utils.pipe_utils import find_trajectory_files

    lag_times = config.lag_times if isinstance(config.lag_times, list) else [config.lag_times]
    stride = config.stride
    timestep_override_ns = getattr(config, 'timestep', None)

    traj_files = find_trajectory_files(config.traj_dir, file_pattern=config.file_pattern)
    if not traj_files:
        sys.exit(f"Error: No trajectory files found in {config.traj_dir} "
                 f"matching '{config.file_pattern}'")

    if timestep_override_ns is not None:
        timestep_ps = timestep_override_ns * 1000.0
        print(f"Using user-specified timestep: {timestep_override_ns} ns ({timestep_ps:.0f} ps)")
    else:
        try:
            traj_iter = md.iterload(traj_files[0], top=config.top, chunk=2)
            first_chunk = next(traj_iter)
            if len(first_chunk.time) < 2:
                print("Warning: Could not infer timestep (need at least 2 frames). "
                      "Skipping lag time validation.")
                return
            timestep_ps = float(first_chunk.time[1] - first_chunk.time[0])
        except Exception as e:
            print(f"Warning: Could not infer timestep from trajectory: {e}. "
                  f"Skipping lag time validation.")
            return

        # Sanity check: warn if inferred timestep seems unreasonable
        timestep_ns_inferred = timestep_ps / 1000.0
        _warn_if_timestep_unreasonable(timestep_ns_inferred, len(traj_files))

    effective_timestep_ps = timestep_ps * stride
    effective_timestep_ns = effective_timestep_ps / 1000.0

    invalid = []
    for lag in lag_times:
        lag_ps = lag * 1000.0
        remainder = lag_ps % effective_timestep_ps
        if remainder > 1e-6 and abs(remainder - effective_timestep_ps) > 1e-6:
            closest = round(lag_ps / effective_timestep_ps) * effective_timestep_ns
            invalid.append((lag, closest))

    if invalid:
        msg = (f"Error: Some lag times are not compatible with the trajectory.\n"
               f"  Trajectory timestep: {timestep_ps:.0f} ps ({effective_timestep_ns:.1f} ns)\n"
               f"  Stride: {stride}\n"
               f"  Effective timestep: {effective_timestep_ps:.0f} ps ({effective_timestep_ns:.1f} ns)\n"
               f"\n"
               f"  Invalid lag times:\n")
        for lag, closest in invalid:
            msg += f"    {lag} ns  ->  closest valid: {closest:.1f} ns\n"
        msg += (f"\n"
                f"  Valid lag times must be multiples of {effective_timestep_ns:g} ns.\n"
                f"  Example valid values: "
                + ", ".join(f"{effective_timestep_ns * i:g}" for i in range(1, 6))
                + ", ...")
        sys.exit(msg)

    print(f"Lag time validation passed: {lag_times} "
          f"(timestep={effective_timestep_ns:.1f} ns, stride={stride})")


def _print_dry_run_summary(config, args):
    """Print a summary of what the pipeline would do without running anything."""
    from pygv.utils.pipe_utils import find_trajectory_files

    print("\n" + "=" * 60)
    print("DRY RUN — Pipeline Preview")
    print("=" * 60)

    # Data
    traj_files = find_trajectory_files(config.traj_dir, file_pattern=config.file_pattern)
    print(f"\nData:")
    print(f"  Topology:          {config.top}")
    print(f"  Trajectory dir:    {config.traj_dir}")
    print(f"  File pattern:      {config.file_pattern}")
    print(f"  Trajectories found: {len(traj_files)}")
    for f in traj_files[:5]:
        print(f"    {f}")
    if len(traj_files) > 5:
        print(f"    ... and {len(traj_files) - 5} more")

    # Configuration
    print(f"\nConfiguration:")
    print(f"  Preset:            {getattr(args, 'preset', None) or 'custom'}")
    print(f"  Encoder:           {config.encoder_type}")
    print(f"  Selection:         {config.selection}")
    print(f"  Stride:            {config.stride}")
    print(f"  Continuous:        {config.continuous}")
    print(f"  N neighbors:       {config.n_neighbors}")
    print(f"  Epochs:            {config.epochs}")
    print(f"  Device:            {'CPU' if config.cpu else 'CUDA if available'}")
    print(f"  Cache:             {config.cache}")
    print(f"  Hurry mode:        {config.hurry}")

    # Planned experiments
    lag_times = config.lag_times if isinstance(config.lag_times, list) else [config.lag_times]
    n_states_list = config.n_states_list if hasattr(config, 'n_states_list') else [config.n_states]
    total = len(lag_times) * len(n_states_list)

    print(f"\nPlanned experiments ({total} total):")
    print(f"  Lag times:         {lag_times}")
    print(f"  N states:          {n_states_list}")
    for lag in lag_times:
        for ns in n_states_list:
            print(f"    lag{lag:g}ns_{ns}states")

    # State discovery
    discover = getattr(config, 'discover_states', True)
    print(f"\nState discovery:     {'enabled' if discover else 'disabled'}")
    if discover:
        print(f"  Min states:        {getattr(config, 'min_states', 2)}")
        print(f"  Max states:        {getattr(config, 'max_states', 10)}")

    # Phases
    print(f"\nPhases to run:")
    if args.only_analysis:
        print(f"  [skip] Preparation")
        print(f"  [skip] Training")
        print(f"  [ ok ] Analysis")
    else:
        print(f"  [{'skip' if args.skip_preparation else ' ok '}] Preparation")
        print(f"  [{'skip' if args.skip_training else ' ok '}] Training")
        print(f"  [ ok ] Analysis")

    # Output
    print(f"\nOutput directory:    {config.output_dir}")
    if args.resume:
        print(f"  Resuming from:     {args.resume}")

    print(f"\n{'=' * 60}")
    print("No actions taken. Remove --dry_run to execute.")
    print("=" * 60)


def main():
    """Main entry point for pipeline"""
    args = parse_pipeline_args()

    # Validate topology file format early
    validate_topology_file(args.top)

    # Load configuration
    if args.preset:
        print(f"Loading preset: {args.preset}")
        config = get_config(args.preset)
    elif args.model:
        print(f"Using model: {args.model}")
        config = get_config(args.model)
    else:
        print("Using base configuration")
        from pygv.config import BaseConfig
        config = BaseConfig()

    # Override with command-line arguments
    config.traj_dir = args.traj_dir
    config.top = args.top
    config.lag_times = args.lag_times
    config.cache = args.cache
    config.hurry = args.hurry
    config.output_dir = args.output_dir
    config.protein_name = args.protein_name
    config.cpu = args.cpu

    # n_states: if explicitly provided, use it and skip discovery for state count
    if args.n_states is not None:
        config.n_states_list = args.n_states
        config._n_states_from_cli = True
    else:
        # Will be determined by state discovery; fall back to config default
        config.n_states_list = [config.n_states]

    # State discovery: on by default, disabled with --no_discover_states
    if args.no_discover_states:
        config.discover_states = False
    if args.min_states is not None:
        config.min_states = args.min_states
    if args.max_states is not None:
        config.max_states = args.max_states

    # Override training parameters only if explicitly provided
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.clip_grad is not None:
        config.clip_grad = args.clip_grad
    if args.stride is not None:
        config.stride = args.stride
    if args.selection is not None:
        config.selection = args.selection
    if args.no_continuous:
        config.continuous = False
    if args.timestep is not None:
        config.timestep = args.timestep

    # Reversible training
    if args.reversible:
        config.reversible = True

    # State diagnostics overrides
    if args.population_threshold is not None:
        config.population_threshold = args.population_threshold
    if args.jsd_threshold is not None:
        config.jsd_threshold = args.jsd_threshold

    # Validate lag times against trajectory timestep before any work begins
    if not args.only_analysis:
        validate_lag_times(config)

    # --validate_only: run all validations above, then exit
    if args.validate_only:
        print("\nAll validations passed.")
        return None

    # --dry_run: preview planned experiments without running anything
    if args.dry_run:
        _print_dry_run_summary(config, args)
        return None

    # Create orchestrator and run pipeline.
    # Intentionally no return: setuptools wraps this as sys.exit(main()), so
    # returning a dict would exit 1 and dump the dict repr to stderr.
    orchestrator = PipelineOrchestrator(config)
    orchestrator.run_complete_pipeline(
        skip_preparation=args.skip_preparation,
        skip_training=args.skip_training,
        only_analysis=args.only_analysis,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
