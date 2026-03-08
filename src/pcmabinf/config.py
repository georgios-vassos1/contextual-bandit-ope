from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExperimentConfig:
    """Global configuration for an OPE experiment."""

    data_dir: Path = field(default_factory=lambda: Path.home() / "pcmabinf" / "OpenML-CC18")
    batch_count: int = 100
    batch_size: int = 100
    reward_variance: float = 1.0
    epsilon_multiplier: float = 0.01
    simulation_count: int = 64
    n_jobs: int = -1
    output_dir: Path = field(default_factory=lambda: Path("results"))
