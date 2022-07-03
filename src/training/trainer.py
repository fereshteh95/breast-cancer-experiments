from ..model_factory import ModelBuilderBase, PatchModelBuilder, WholeModelBuilder
from .base import TrainerBase

import mlflow
import typing


class Trainer(TrainerBase):

    def __init__(self, config, run_dir, exported_dir) -> None:
        super().__init__(config, run_dir, exported_dir)
        self. initial_learning_rate = self.config.info_training.initial_learning_rate
        self. activation = self.config.general_info.activation
        self. image_dimension = self.config.general_info.image_dimension
        self.loss = self.config.info_training.loss
        self. epochs = self.config.info_training.epochs

    def train(self,
              model: ModelBuilderBase,
              train_data_gen,
              n_iter_train,
              val_data_gen,
              n_iter_val,
              class_weight,
              callbacks,
              active_run: typing.Optional[mlflow.ActiveRun] = None):
        """
        Training the model.
        """
        with mlflow.start_run(nested=True):

            history = model.fit(
                train_data_gen,
                steps_per_epoch=n_iter_train,
                epochs=self.epochs,
                validation_data=val_data_gen,
                validation_steps=n_iter_val,
                callbacks=callbacks,
                class_weight=class_weight

            )

    def export(self):

        """
         Exports the best model to `self.exported_dir`, logs the model to mlflow's model registry,
         and adds the model's address to the config file to be versioned using git.
         """
        pass
        
     
        
