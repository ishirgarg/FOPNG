from typing import Any, Dict, Optional
from dataclasses import dataclass, asdict
import torch

@dataclass
class Config:
    """Global experiment configuration."""
    seed: int
    batch_size: int
    lr: float
    epochs_per_task: int
    grads_per_task: int
    device: str
    
    # Logging
    log_dir: Optional[str]
    save_model: bool
    save_plots: bool
    save_raw_data: bool
    experiment_name: Optional[str] = "noname"

    # FOPNG specific
    fopng_lambda_reg: float = 0.0
    fopng_epsilon: float = 0.0
    
    def __post_init__(self):
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {k: v for k, v in asdict(self).items() if not k.startswith('_')}