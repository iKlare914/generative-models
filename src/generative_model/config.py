from pydantic import BaseModel, ConfigDict, PositiveInt, PositiveFloat, Field
from pathlib import Path

class StrictModel(BaseModel):
    """Base configuration model that rejects undeclared fields."""

    model_config = ConfigDict(extra='forbid')

class TrainerConfig(StrictModel, frozen=True):
    """Training configuration whose fields cannot be reassigned after creation.

    Attributes:
        lr: Positive optimizer learning rate.
        batch_size: Number of samples per training batch. Must be positive.
        dropout: dropout rate for model. Must be positive float
        weight_decay: Weight decay coefficient for the optimizer.
        epoches: Total number of training epochs. Must be a positive integer.
        save_interval_epoch: Interval for saving status. Must be positive integer
        resume: Whether to resume training from an existing checkpoint.
            Defaults to False.
        guided: Whether to use guided diffusion
        log_samples: Whether to log samples result every save interval epoches
        save_dir: Directory in which to save training checkpoints.
    """

    lr: PositiveFloat
    batch_size: PositiveInt
    dropout: float = Field(ge=0, le=1)
    weight_decay: float = Field(ge=0)
    epoches: PositiveInt
    save_interval_epoch: PositiveInt
    resume: bool = False
    guided: bool = False
    log_samples: bool = False
    save_dir: Path
    image_size: PositiveInt = 32
