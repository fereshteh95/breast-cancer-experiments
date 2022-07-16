from abc import ABC, abstractmethod
from pathlib import Path
import typing

import mlflow
from omegaconf.dictconfig import DictConfig

from model_factory import ModelBuilderBase


class TrainerBase(ABC):
    
    """Responsibilities:
        - training
        - exporting
        - generating tensorboard and mlflow training metrics
        - resume training if interrupted

    Attributes:

        - config: config file
        - run_dir: where to write the checkpoints
        - exported_dir: where to write the exported

    Examples:

        >>> from pathlib import Path
        >>> from omegaconf import OmegaConf
        >>> from mlflow import ActiveRun
        >>> config = OmegaConf.load('./config.yaml')
        >>> run_dir = Path('run')
        >>> run_dir.mkdir(exist_ok=True)
        >>> exported_dir = Path('exported')
        >>> trainer = TrainerBase(config, run_dir, exported_dir)
        >>> trainer.train(...)


    """

    def __init__(self,
                 config: DictConfig,
                 run_dir: Path,
                 exported_dir: Path):

        self.config = config
        self.run_dir = run_dir
        self.exported_dir = exported_dir

    @abstractmethod
    def train(self,
              model: ModelBuilderBase,
              train_data_gen,
              n_iter_train,
              val_data_gen,
              n_iter_val,
              class_weight,
              callbacks,
              active_run: typing.Optional[mlflow.ActiveRun] = None):
        """Training the model."""

    # @abstractmethod
    # def export(self):
    #     """
    #      Exports the best model to `self.exported_dir`, logs the model to mlflow's model registry,
    #      and adds the model's address to the config file to be versioned using git.
    #     """
