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
    
    # Wandb configuration
    wandb_project: Optional[str] = "fopng-experiments"
    wandb_entity: Optional[str] = None
    wandb_tags: Optional[list] = None
    use_wandb: bool = True  # Set to False to disable wandb logging

    # FOPNG specific
    fopng_lambda_reg: float = 0.0
    fopng_new_fisher_weight: float = 0.5  # Weight for new Fisher: F_old = (1-w)*F_old + w*F_current
    use_empirical_fisher: bool = False  # For FOPNG-PF: compute F*g on-the-fly during gradient collection instead of pre-multiplying by estimated Fisher
    fisher_batch_size: Optional[int] = None  # If set, estimate Fisher from this batch size instead of using the full training set
    ewc_lambda: float = 100.0
    first_task_lr: Optional[float] = None  # If set, use this learning rate for the first task (task_id=0) instead of lr
    use_adam: bool = False  # If True, use Adam optimizer for first task instead of SGD (or whatever optimizer is passed)
    use_sgd: bool = False  # If True, use SGD optimizer for first task instead of Adam (or whatever optimizer is passed)
    
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
        # Filter out logging-related config that shouldn't be logged as metrics
        exclude_keys = {'log_dir', 'save_model', 'save_plots', 'save_raw_data', 
                       'wandb_project', 'wandb_entity', 'wandb_tags', 'use_wandb'}
        return {k: v for k, v in asdict(self).items() 
                if not k.startswith('_') and k not in exclude_keys}