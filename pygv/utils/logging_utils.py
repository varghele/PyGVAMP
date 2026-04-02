#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Logging utilities for PyGVAMP pipeline.

Provides a TeeStream that duplicates stdout/stderr to a log file,
so all console output is automatically saved.
"""

import sys
import os
from datetime import datetime


class TeeStream:
    """
    A stream wrapper that writes to both a file and the original stream.

    This allows all print() output to be captured in a log file while
    still displaying on the console.
    """

    def __init__(self, log_file, original_stream):
        self.log_file = log_file
        self.original_stream = original_stream

    def write(self, message):
        self.original_stream.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.original_stream.flush()
        self.log_file.flush()

    def fileno(self):
        return self.original_stream.fileno()

    def isatty(self):
        return self.original_stream.isatty()


class PipelineLogger:
    """
    Manages log file creation and stdout/stderr redirection.

    Usage
    -----
    >>> logger = PipelineLogger(log_dir="/path/to/logs")
    >>> logger.start()
    >>> # ... all print() output is now also written to the log file ...
    >>> logger.stop()

    Or as a context manager:
    >>> with PipelineLogger(log_dir="/path/to/logs") as logger:
    ...     print("This goes to console and log file")
    """

    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.log_file = None
        self.log_path = None
        self._original_stdout = None
        self._original_stderr = None

    def start(self):
        """Start logging by redirecting stdout and stderr to a tee stream."""
        os.makedirs(self.log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(self.log_dir, f"log_{timestamp}.txt")
        self.log_file = open(self.log_path, "w")

        # Write header
        self.log_file.write(f"PyGVAMP Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log_file.write(f"Command: {' '.join(sys.argv)}\n")
        self.log_file.write("=" * 60 + "\n\n")
        self.log_file.flush()

        # Replace stdout and stderr
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = TeeStream(self.log_file, self._original_stdout)
        sys.stderr = TeeStream(self.log_file, self._original_stderr)

        return self

    def stop(self):
        """Stop logging and restore original stdout/stderr."""
        if self._original_stdout is not None:
            sys.stdout = self._original_stdout
            self._original_stdout = None
        if self._original_stderr is not None:
            sys.stderr = self._original_stderr
            self._original_stderr = None
        if self.log_file is not None:
            self.log_file.write("\n" + "=" * 60 + "\n")
            self.log_file.write(f"Log ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.log_file.close()
            self.log_file = None

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
