"""

    inference_engine, contains the shared-preprocessing + preprocessing + inference + post-processing

"""
from ..prepare import Preparation
from .base import ModelBuilderBase

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50


class PatchModelBuilder(ModelBuilderBase):

    """
    This class creates a classification model based on the intended backbone. 
    It takes the config file and the intended backbone as inputs and adds a dense layer on top of that.You can either
    add a dropout layer with  intended dropout_rate or  not. for training, the 
    phase argument must be set as phase = train in order to get  metrics   and 
    perform model compiling.

    Example:
        model_builder = model(config, MobileNet, phase = 'train')
        compiled_model = model_builder . get_model()
        callbacks = model_builder. get_callbacks()
    """
    def __init__(self, config, dropout=True, lr=1e-5, dropout_rate=.3, phase=None) -> None:
        super().__init__(config)
        self.backbone = ResNet50
        self. class_names = self.config.general_info.classes
        self.input_shape = (self.config.general_info.image_height_patch,
                            self.config.general_info.image_width_patch,
                            self.config.general_info.image_channels
                            )
        self.classes = len(self. class_names)
        self.activation = self.config.general_info.activation
        self.loss = self.config.info_training.loss
        self.dropout = dropout
        self.dropout_rate = dropout_rate
        self.lr = lr
        self.phase = phase
        self. history_output_path = '../' + self.config.info_training.history_output_path
        self.output_weights_name = '../' + self.config.info_training.output_weights_name

    def get_model(self, hp=None) -> tf.keras.Model:
        """Generates the model for training, and returns the compiled model.

        Returns:
            A compiled ``tensorflow.keras`` model.
        """
        img_input = Input(shape=self.input_shape)
        base_model = self.backbone(
            include_top=False,
            input_tensor=img_input,
            input_shape=self.input_shape,
            weights='imagenet',
            pooling="avg"
        )
        x = base_model.output
        if self.dropout:
            x = Dropout(self.dropout_rate)(x)
        predictions = Dense(self.classes, activation=self.activation, name="new_predictions")(x)

        m = Model(inputs=img_input, outputs=predictions)
        if self.phase == 'train':
            prepare = Preparation()
            metrics_all = prepare.metrics_define(len(self.class_names))
            optimizer = Adam(learning_rate=self.lr)
            m.compile(optimizer=optimizer, loss=self.loss, metrics=metrics_all)
        elif self.phase == 'evaluation':
            m.load_weights(self.config.general_info.best_weights_path)
        return m

    def get_callbacks(self) -> list:
        """Returns any callbacks for ``fit``.

        Returns:
            list of ``tf.keras.Callback`` objects. ``Orchestrator`` will handle the ``ModelCheckpoint`` and ``Tensorboard`` callbacks.
            Still, you can return each of these two callbacks, and orchestrator will modify your callbacks if needed.

        """
        check1 = ModelCheckpoint(self.output_weights_name,
                                 monitor='val_accuracy',
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=True,
                                 mode='max')

        lr_reduction = ReduceLROnPlateau(monitor='val_accuracy', factor=0.9, patience=1, verbose=1, mode="max",
                                         min_lr=1e-8)
        history_logger = CSVLogger(self. history_output_path, separator=",", append=True)
        return [check1, lr_reduction, history_logger]
