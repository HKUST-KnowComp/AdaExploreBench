"""Benchmark execution and sequential runtime.

Keep package imports light: dataset and runtime modules may require optional
CLI dependencies such as PyYAML, while lower-level consumers often only need
mechanism helpers.
"""

from protocol.mechanism import FAMILIES

__all__ = ["FAMILIES"]
