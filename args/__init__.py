# args/__init__.py

"""
Command line argument handling for PyGVAMP
"""

from .args_train import parse_train_args, get_train_parser

__all__ = ['parse_train_args', 'get_train_parser']
