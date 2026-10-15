"""Dependency-driven synthesis of canonical AdaExploreBench domain templates."""

from .config import SynthesisConfig, load_config
from .pipeline import TemplateSynthesisPipeline

__all__ = ["SynthesisConfig", "TemplateSynthesisPipeline", "load_config"]
