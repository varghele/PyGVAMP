# args/__init__.py

"""
Command line argument handling for PyGVAMP.

This module provides argument parsers for all pipeline phases:
- Base arguments (common to all phases)
- Encoder arguments (SchNet, Meta, ML3)
- Training arguments
- Preparation arguments
- Analysis arguments
- Full pipeline arguments
"""

# Base argument functions
from .args_base import (
    add_common_args,
    add_data_args,
    add_processing_args,
    add_graph_args,
    add_output_args,
    add_hardware_args,
    get_base_parser,
)

# Encoder argument functions
from .args_encoder import (
    add_encoder_args,
    add_encoder_selection_args,
    add_schnet_args,
    add_meta_args,
    add_ml3_args,
)

# Training argument functions
from .args_train import (
    parse_train_args,
    get_train_parser,
    add_embedding_args,
    add_classifier_args,
    add_training_args,
)

# Preparation argument functions
from .args_prep import (
    parse_prep_args,
    get_prep_parser,
    add_prep_args,
)

# Analysis argument functions
from .args_anly import (
    parse_anly_args,
    get_anly_parser,
)

# Pipeline argument functions
from .args_pipeline import (
    parse_pipeline_args,
    get_pipeline_parser,
    add_pipeline_control_args,
    add_grid_search_args,
)

__all__ = [
    # Base
    'add_common_args',
    'add_data_args',
    'add_processing_args',
    'add_graph_args',
    'add_output_args',
    'add_hardware_args',
    'get_base_parser',
    # Encoder
    'add_encoder_args',
    'add_encoder_selection_args',
    'add_schnet_args',
    'add_meta_args',
    'add_ml3_args',
    # Training
    'parse_train_args',
    'get_train_parser',
    'add_embedding_args',
    'add_classifier_args',
    'add_training_args',
    # Preparation
    'parse_prep_args',
    'get_prep_parser',
    'add_prep_args',
    # Analysis
    'parse_anly_args',
    'get_anly_parser',
    # Pipeline
    'parse_pipeline_args',
    'get_pipeline_parser',
    'add_pipeline_control_args',
    'add_grid_search_args',
]
