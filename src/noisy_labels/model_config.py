from dataclasses import dataclass
from typing import Literal


@dataclass
class ModelConfig:
    dataset_name: Literal["A", "B", "C", "D"]
    batch_size: int = 64
    hidden_dim: int = 128
    latent_dim: int = 8
    num_classes: int = 6
    epochs: int = 1000
    learning_rate: float = 0.0005
    num_cycles: int = 5
    warmup: int = 5
    early_stopping_patience: int = 100
