# args/__init__.py

"""
Command line argument handling for PyGVAMP
"""

from .args_train import parse_train_args, get_train_parser
from .args_prep import parse_prep_args, get_prep_parser
from .args_anly import parse_anly_args, get_anly_parser

__all__ = [
    'parse_train_args',
    'get_train_parser',
    'parse_prep_args',
    'get_prep_parser',
    'parse_anly_args',
    'get_anly_parser',
]
