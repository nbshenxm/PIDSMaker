"""Velox-specific PIDS implementation package.

This package is intended to hold the actual Velox model, encoder, objectives,
and trainer code so that the system can be modified independently of the generic
shared factory-based infrastructure.
"""

from .trainer import VeloxTrainer

__all__ = ["VeloxTrainer"]
