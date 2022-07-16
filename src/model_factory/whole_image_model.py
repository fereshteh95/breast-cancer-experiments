import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from classification_models.tfkeras import Classifiers

from prepare import Preparation
from .base import ModelBuilderBase


class WholeModelBuilder(ModelBuilderBase):
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

    def __init__(self, config, dropout=True, dropout_rate=.3, phase=None) -> None:
        super().__init__(config)
        self.class_names = self.config.general_info.classes
        self.input_shape = (self.config.general_info.image_height,
                            self.config.general_info.image_width,
                            self.config.general_info.image_channels
                            )
        self.classes = len(self.class_names)
        self.base_model_name = self.config.general_info.base_model_name
        self.activation = self.config.general_info.activation
        self.loss = self.config.info_training.loss
        self.dropout = dropout
        self.dropout_rate = dropout_rate
        self.lr = self.config.info_training.initial_learning_rate
        self.phase = phase

    def get_model(self, hp=None) -> tf.keras.Model:
        """Generates the model for training, and returns the compiled model.

        Returns:
            A compiled ``tensorflow.keras`` model.
        """
        basemodel, preprocess_input = Classifiers.get(self.base_model_name)
        engine = basemodel(include_top=False,
                           input_shape=self.input_shape,
                           backend=tf.keras.backend,
                           layers=tf.keras.layers,
                           models=tf.keras.models,
                           utils=tf.keras.utils,
                           weights='imagenet')

        x = GlobalAveragePooling2D(name='avg_pool')(engine.output)
        if self.dropout:
            x = Dropout(self.dropout_rate)(x)
        x = Dense(256, activation='relu', name='DenseLayer')(x)
        predictions = Dense(self.classes, activation=self.activation, name="new_predictions")(x)
        image_model = Model(inputs=engine.input, outputs=predictions)
        if self.phase == 'train':
            prepare = Preparation()
            metrics_all = prepare.metrics_define(len(self.class_names))
            optimizer = Adam(learning_rate=self.lr)
            image_model.compile(optimizer=optimizer, loss=self.loss, metrics=metrics_all)
        elif self.phase == 'evaluation':
            image_model = image_model
        return image_model

    def get_callbacks(self) -> list:
        """Returns any callbacks for ``fit``.

        Returns:
            list of ``tf.keras.Callback`` objects. ``Orchestrator`` will handle the ``ModelCheckpoint`` and ``Tensorboard`` callbacks.
            Still, you can return each of these two callbacks, and orchestrator will modify your callbacks if needed.

        """

        lr_reduction = ReduceLROnPlateau(monitor=self.config.info_training.export_metric,
                                         factor=0.7,
                                         patience=3,
                                         verbose=1,
                                         mode=self.config.info_training.export_mode,
                                         min_lr=1e-8)
        return [lr_reduction]
