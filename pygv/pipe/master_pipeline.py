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
import argparse
import json
from datetime import datetime
from pathlib import Path

# Import CacheManager
from pygv.pipe.caching import CacheManager

# Import pipeline components
from pygv.pipe.preparation import run_preparation
from pygv.pipe.training import run_training
from pygv.pipe.analysis import run_analysis
from pygv.config import get_config, list_presets
from pygv.pipe.args import parse_pipeline_args

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
        self.cache_manager = CacheManager(config)

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

        # Check if cached dataset exists
        dataset_hash = self.cache_manager.get_dataset_hash()
        cached_dataset = self.cache_manager.check_cached_dataset(dataset_hash)

        if cached_dataset and self.config.cache:
            print(f"Found cached dataset: {cached_dataset}")
            return cached_dataset, None

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

        # Cache dataset if requested
        if self.config.cache:
            self.cache_manager.cache_dataset(dataset_path, dataset_hash)

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

                # Check if model already exists
                model_path = exp_dir / 'best_model.pt'
                if model_path.exists():
                    print(f"Model already exists: {model_path}")
                    trained_models[exp_name] = str(model_path)
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

        # Infer timestep from trajectory (in picoseconds)
        try:
            # Load first few frames to get timestep
            import mdtraj as md
            traj_iterator = md.iterload(self.config.traj_dir, top=self.config.top, chunk=2)
            first_chunk = next(traj_iterator)

            if len(first_chunk.time) < 2:
                raise ValueError("Trajectory must have at least 2 frames to infer timestep")

            # Calculate timestep from time differences (in picoseconds)
            timestep_ps = first_chunk.time[1] - first_chunk.time[0]

            # Convert to nanoseconds for comparison with lag_time
            timestep_ns = timestep_ps / 1000.0

            print(f"Inferred trajectory timestep: {timestep_ps:.3f} ps ({timestep_ns:.6f} ns)")

        except Exception as e:
            print(f"Warning: Could not infer timestep from trajectory: {str(e)}")
            print("Falling back to assumed timestep of 0.1 ns")
            timestep_ns = 0.1  # Fallback to assumed value

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

        # Get lag_time from model path
        import re
        match = re.search(r'lag(\d+)ns', str(model_path))
        if match:
            args.lag_time = float(match.group(1))
        else:
            args.lag_time = self.config.lag_times[0] if isinstance(self.config.lag_times,
                                                                   list) else self.config.lag_time

        return args

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

    def run_complete_pipeline(self):
        """Run the complete pipeline"""
        print("=" * 60)
        print("STARTING PYGVAMP PIPELINE")
        print("=" * 60)
        print(f"Protein: {self.config.protein_name}")
        print(f"Model: {self.config.encoder_type}")
        print(f"Preset: {getattr(self.config, 'preset', 'custom')}")
        print(f"Cache: {self.config.cache}")
        print(f"Hurry mode: {self.config.hurry}")

        # Setup experiment directory
        dirs = self.setup_experiment_directory()

        # Phase 1: Preparation
        dataset_path, recommended_n_states = self.run_preparation_phase(dirs)

        # Use discovered n_states if no explicit list was provided
        if recommended_n_states is not None and not hasattr(self.config, '_n_states_from_cli'):
            print(f"Using discovered n_states = {recommended_n_states}")
            self.config.n_states_list = [recommended_n_states]

        # Phase 2: Training
        trained_models = self.run_training_phase(dirs, dataset_path)

        # Phase 3: Analysis
        analysis_results = self.run_analysis_phase(dirs, trained_models)

        # Save summary
        self._save_pipeline_summary(dirs, trained_models, analysis_results)

        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED")
        print("=" * 60)
        print(f"Results saved to: {self.experiment_dir}")

        return {
            'experiment_dir': self.experiment_dir,
            'trained_models': trained_models,
            'analysis_results': analysis_results
        }

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


def main():
    """Main entry point for pipeline"""
    args = parse_pipeline_args()

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
    if args.clip_grad is not None:
        config.clip_grad = args.clip_grad
    if args.stride is not None:
        config.stride = args.stride
    if args.selection is not None:
        config.selection = args.selection
    if args.no_continuous:
        config.continuous = False

    # Create orchestrator and run pipeline
    orchestrator = PipelineOrchestrator(config)
    results = orchestrator.run_complete_pipeline()

    return results


if __name__ == "__main__":
    results = main()
