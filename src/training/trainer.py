from pathlib import Path
import shutil

from model_factory import ModelBuilderBase
from prepare import Preparation
from .base import TrainerBase
from model_factory import PyfuncModel

import tensorflow as tf
import os
import mlflow
import typing
from omegaconf.dictconfig import DictConfig


class Trainer(TrainerBase):

    def __init__(self, config, run_dir, exported_dir) -> None:
        super().__init__(config, run_dir, exported_dir)
        self.initial_learning_rate = self.config.info_training.initial_learning_rate
        self.activation = self.config.general_info.activation
        self.loss = self.config.info_training.loss
        self.epochs = self.config.info_training.epochs
        self.class_names = self.config.general_info.classes

        self.checkpoints_dir = self.run_dir.joinpath('checkpoints')
        self.tensorboard_log_dir = self.run_dir.joinpath('logs')

        self.checkpoints_dir.mkdir(exist_ok=True, parents=True)
        self.tensorboard_log_dir.mkdir(exist_ok=True, parents=True)
        self.history = None

    def train(self,
              model: ModelBuilderBase,
              train_data_gen,
              n_iter_train,
              val_data_gen,
              n_iter_val,
              class_weight,
              callbacks,
              active_run: typing.Optional[mlflow.ActiveRun] = None,
              retrain: bool = False
              ):
        """
        Training the model.
        """

        print(f'available GPU devices: {tf.config.list_physical_devices()}')

        with active_run as active_run:
            run_id = active_run.info.run_id

            mlflow.log_param('base_model_name', self.config.mlflow.model_name)
            mlflow.log_param('training_mode', self.config.mlflow.training_mode)
            mlflow.log_param('num_epochs', self.config.info_training.epochs)
            mlflow.log_param('train_batch_size', self.config.general_info.train_batch_size)
            mlflow.log_param('validation_batch_size', self.config.general_info.val_batch_size)
            mlflow.log_param('image_height', self.config.general_info.image_height)
            mlflow.log_param('image_width', self.config.general_info.image_width)
            mlflow.log_param('image_channels', self.config.general_info.image_channels)
            mlflow.log_param('learning_rate', self.config.info_training.initial_learning_rate)
            mlflow.log_param('dataset_name', self.config.mlflow.dataset)

            if retrain or not any(_get_checkpoints(self.checkpoints_dir)):
                shutil.rmtree(self.checkpoints_dir, ignore_errors=True)
                shutil.rmtree(self.tensorboard_log_dir, ignore_errors=True)

                self.checkpoints_dir.mkdir(exist_ok=True, parents=True)
                self.tensorboard_log_dir.mkdir(exist_ok=True, parents=True)

                initial_epoch = 0
                model = model
                initial_threshold = None
            else:
                model, initial_epoch, initial_threshold = self._load_latest_model()
                mlflow.set_tag('session_type', 'resumed_training')

            callbacks = self._get_callbacks(callbacks,  active_run, initial_threshold)

            history = model.fit(
                train_data_gen,
                steps_per_epoch=n_iter_train,
                epochs=self.epochs,
                validation_data=val_data_gen,
                validation_steps=n_iter_val,
                callbacks=callbacks,
                class_weight=class_weight

            )
            self.history = history

        # pyfuncmodel = PyfuncModel()
        # exporter = Exporter(self.config, self.run_dir)
        # exporter.log_model_to_mlflow(active_run,
        #                              pyfuncmodel,
        #                              Path('../config.yaml')
        #                              )

    def _write_mlflow_run_id(self, run: mlflow.ActiveRun):
        run_id_path = self.run_dir.joinpath('run_id.txt')
        with open(run_id_path, 'w') as f:
            f.write(run.info.run_id)

    def _get_callbacks(self,
                       callbacks,
                       active_run: mlflow.ActiveRun,
                       initial_threshold=None):
        """Makes sure that TensorBoard and ModelCheckpoint callbacks exist and are correctly configured.

        Attributes:
            model_builder: ``ModelBuilder`` object, to get callbacks list using ``model_builder.get_callbacks``

        modifies ``callbacks`` to be a list of callbacks, in which ``TensorBoard`` callback exists with
         ``log_dir=self.tensorboard_log_dir`` and ``ModelCheckpoint`` callback exists with
          ``filepath=self.checkpoints_dir/...``, ``save_weights_only=False``

        """

        class MLFlowLogging(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                mlflow.log_metrics(logs, epoch)

        mc_callbacks = [i for i in callbacks if isinstance(i, tf.keras.callbacks.ModelCheckpoint)]
        tb_callbacks = [i for i in callbacks if isinstance(i, tf.keras.callbacks.TensorBoard)]

        to_track = self.config.info_training.export_metric
        checkpoint_path = str(self.checkpoints_dir) + "/densenet121-image-classifier-benign-malignant"
        # checkpoint_path = checkpoint_path + "-{" + to_track + ":4.5f}"

        if any(mc_callbacks):
            mc_callbacks[0].filepath = str(checkpoint_path)
            mc_callbacks[0].save_weights_only = False
        else:
            mc = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=False,
                                                    monitor=to_track,
                                                    verbose=1,
                                                    save_best_only=True,
                                                    initial_value_threshold=initial_threshold
                                                    )
            callbacks.append(mc)

        if any(tb_callbacks):
            tb_callbacks[0].log_dir = self.tensorboard_log_dir
        else:
            tb = tf.keras.callbacks.TensorBoard(log_dir=self.tensorboard_log_dir)
            callbacks.append(tb)

        mlflow_logging_callback = MLFlowLogging()
        callbacks.append(mlflow_logging_callback)

        return callbacks

    def _load_latest_model(self):
        """Loads and returns the latest ``SavedModel``.

        Returns:
            model: a ``tf.keras.Model`` object.
            initial_epoch: initial epoch for this checkpoint

        """

        latest_ch = self._get_latest_checkpoint()
        initial_epoch = latest_ch['epoch']
        initial_threshold = latest_ch['value']
        sm_path = latest_ch['path']
        print(f'found latest checkpoint: {sm_path}')
        print(f'resuming from epoch {initial_epoch}')
        model = tf.keras.models.load_model(latest_ch['path'])
        return model, initial_epoch, initial_threshold

    def _get_latest_checkpoint(self):
        """Returns info about the latest checkpoint.

        Returns:
            a dictionary containing epoch, path to ``SavedModel`` and value of ``self.export_metric`` for
            latest checkpoint:
                {'epoch': int, 'path': pathlib.Path, 'value': float}

        """
        checkpoints = get_checkpoints_info(self.checkpoints_dir)

        if self.config.info_training.export_mode == 'min':
            selected_model = min(checkpoints, key=lambda x: x['value'])
        else:
            selected_model = max(checkpoints, key=lambda x: x['value'])

        return selected_model


class Exporter:
    """Exports the best checkpoint as a `mlflow.pyfunc`, """

    def __init__(self, config: DictConfig, run_dir: Path, exported_dir: Path = Path('../exported')):
        self.config = config

        self.exported_dir = Path(self.config.info_training.export)

        self.checkpoints_dir = run_dir.joinpath('checkpoints')
        self.tensorboard_log_dir = run_dir.joinpath('logs')
        self.exported_model_path = exported_dir.joinpath('savedmodel')

    def log_model_to_mlflow(self,
                            active_run: mlflow.ActiveRun,
                            pyfunc_model: mlflow.pyfunc.PythonModel,
                            config_path: Path,
                            signature: typing.Optional[mlflow.models.ModelSignature] = None,
                            mlflow_pyfunc_model_path: str = "tfsm_mlflow_pyfunc"):
        """Logs the best model from `self.checkpoints_dir` to the given active_run as an artifact.

        Notes:
            - you can load and use the model by `loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)`
        """

        best_model_info = self.get_best_checkpoint()
        print(f'exporting the {best_model_info["path"]} ..')

        with active_run as _:
            artifacts = {
                "savedmodel_path": str(best_model_info['path']),
                "config_path": str(config_path)
            }

            model_info = mlflow.pyfunc.log_model(
                artifact_path=mlflow_pyfunc_model_path,
                python_model=pyfunc_model,
                artifacts=artifacts,
                signature=signature,
                code_path=None)

        return model_info

    def export(self) -> Path:
        """Exports the best version of ``SavedModel`` s, and ``config.yaml`` file into exported sub_directory.

        This method will delete all checkpoints after exporting the best one.
        """

        self.check_for_exported()

        best_model_info = self.get_best_checkpoint()

        # exported_config_path = self.initial_export_dir.joinpath('config.yaml')
        shutil.copytree(best_model_info['path'], self.exported_model_path,
                        symlinks=False, ignore=None, ignore_dangling_symlinks=False)
        # self._write_dict_to_yaml(dict_config, exported_config_path)

        # Delete checkpoints
        shutil.rmtree(self.checkpoints_dir)
        return self.exported_model_path

    def check_for_exported(self):
        """Raises exception if exported directory exists and contains ``savedmodel``"""

        if self.exported_dir.is_dir():
            if any(self.exported_dir.iterdir()):
                if self.exported_model_path.exists():
                    raise Exception('exported savedmodel already exist.')

    def get_best_checkpoint(self):
        """Returns info about the best checkpoint.

        Returns:
            a dictionary containing epoch, path to ``SavedModel`` and value of ``self.export_metric`` for
            the best checkpoint in terms of ``self.export_metric``:
                {'epoch': int, 'path': pathlib.Path, 'value': float}

        """

        checkpoints = get_checkpoints_info(self.checkpoints_dir)

        if self.config.info_training.export_mode == 'min':
            selected_model = min(checkpoints, key=lambda x: x['value'])
        else:
            selected_model = max(checkpoints, key=lambda x: x['value'])
        return selected_model


def get_checkpoints_info(checkpoints_dir: Path):
    """Returns info about checkpoints.

    Returns:
        A list of dictionaries related to each checkpoint:
            {'epoch': int, 'path': pathlib.Path, 'value': float}

    """

    checkpoints = _get_checkpoints(checkpoints_dir)
    ckpt_info = list()
    for cp in checkpoints:
        splits = str(cp.name).split('-')
        epoch = int(splits[1])
        metric_value = float(splits[2])
        ckpt_info.append({'path': cp, 'epoch': epoch, 'value': metric_value})
    return ckpt_info


def _get_checkpoints(checkpoints_dir: Path):
    """Returns a list of paths to folders containing a ``saved_model.pb``"""

    ckpts = [item for item in checkpoints_dir.iterdir() if any(item.glob('saved_model.pb'))]
    return ckpts
