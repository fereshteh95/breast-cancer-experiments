from abc import ABC, abstractmethod

import tensorflow.keras as tfk
from omegaconf.dictconfig import DictConfig


class ModelBuilderBase(ABC):

    """Building and compiling ``tensorflow.keras`` models to train with Trainer.

        Notes:
            - you have to override these methods: ``get_model``
            - you may override these methods too (optional): ``get_callbacks``, ``get_class_weight``
            - don't override the private ``_{method_name}`` methods

        Examples:
            >>> model_builder = ModelBuilderBase(config)
            >>> model = model_builder.get_model()
            >>> callbacks = model_builder.get_callbacks()
            >>> model.fit(train_gen, n_iter_train, callbacks=callbacks, class_weight=class_weight)

        """
    def __init__(self, config: DictConfig):
        self.config = config

    @abstractmethod
    def get_model(self) -> tfk.Model:
        """Generates the model for training, and returns the compiled model.

        Returns:
            A compiled ``tensorflow.keras`` model.
        """

    @abstractmethod
    def get_callbacks(self) -> list:
        """Returns any callbacks for ``fit``.

        Returns:
            list of ``tf.keras.Callback`` objects. ``Orchestrator`` will handle the ``ModelCheckpoint`` and ``Tensorboard`` callbacks.
            Still, you can return each of these two callbacks, and orchestrator will modify your callbacks if needed.

        """
