"""
MD Visualizer - Interactive 3D Visualization Toolkit for Molecular Dynamics Trajectory Analysis

This package provides tools for creating interactive web-based visualizations of molecular
dynamics trajectories, combining:
- 2D embeddings in 3D space (multi-timescale analysis)
- Protein structure visualization with attention mapping
- Transition matrices and state information

Example usage:
    >>> from md_visualizer import MDTrajectoryVisualizer
    >>> pygviz = MDTrajectoryVisualizer()
    >>> pygviz.add_timescale(
    ...     lagtime=10,
    ...     embeddings=embeddings_2d,
    ...     frame_indices=frame_indices,
    ...     state_assignments=states,
    ...     transition_matrix=trans_matrix,
    ...     attention_values=attention
    ... )
    >>> pygviz.set_protein_structure(pdb_path="protein.pdb")
    >>> pygviz.generate("output.html")
"""

from .visualizer import MDTrajectoryVisualizer
from .data_handler import DataProcessor

__version__ = '0.1.0'
__author__ = 'PyGVAMP Contributors'
__all__ = ['MDTrajectoryVisualizer', 'DataProcessor']
