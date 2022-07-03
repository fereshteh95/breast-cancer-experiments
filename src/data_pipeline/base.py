from abc import ABC, abstractmethod
import typing


class DataLoaderBase(ABC):
    """Data-loading mechanism.

    This class will create data generators.
    This is actually the process of ingesting data from a data source into data generators

    Attributes:
        config (DictConfig): config file


    Examples:
        >>> data_loader = DataLoader(config)
        >>> data_gen, data_n = data_loader.create_training_generator()
        >>> test_data_gen, test_data_n = data_loader.create_test_generator()

    """

    def __init__(self, config):
        self.config = config

    @abstractmethod
    def create_training_generator(self) -> typing.Tuple[typing.Iterator, int]:

        """Create data generator for training sub-set.

        This will create a ``generator``/``tf.data.Dataset`` which yields three components:
        `(input_batch, label_batch, sample_weight(s)_batch)`

        Notes:
            - If you don't need ``sample_weight``, set it to ``1`` for all data-points.

        Returns:
            tuple(generator, n_iter_train):
            - generator: a ``generator``/``tf.data.Dataset``.
            - n_iter_train: number of iterations per epoch for training data generator.

        """

    @abstractmethod
    def get_class_weight(self) -> typing.Optional[dict]:
        """Set this if you want to pass ``class_weight`` to ``fit``.

        Returns:
           Optional dictionary mapping class indices (integers) to a weight (float) value.
           used for weighting the loss function (during training only).
           This can be useful to tell the model to "pay more attention" to samples from an under-represented class.

        """

    @abstractmethod
    def create_validation_generator(self):

        """Create data generator for validation sub-set.

        This will create a ``generator``/``tf.data.Dataset`` which yields three components:
        `(input_batch, label_batch, sample_weight(s)_batch)`

        Notes:
            - If you don't need ``sample_weight``, set it to ``1`` for all data-points.
            - Make sure that the shuffling is off for validation generator.

        Returns:
            tuple(generator, n_iter_val):
            - generator: a ``generator``/``tf.data.Dataset``.
            - n_iter_val: number of iterations per epoch for validation data generator.

        """

    @abstractmethod
    def create_test_generator(self):
        """Create data generator for test sub-set.

        This will create a ``generator``/``tf.data.Dataset`` which yields three components:
        (input, label, data_id(str)),

        Notes:
            - Each ``data_id`` could be anything specific that can help to retrieve this data point.
            - You can consider to set ``data_id=row_id`` of the test subset's dataframe, if you have one.
            - Do not repeat this dataset, i.e. raise an exception at the end.

        Returns:
            tuple(generator, test_n):
            - generator: a ``generator``/``tf.data.Dataset``.
            - test_n: number of test data-points.

        """


class BaseClass(ABC):

    def __init__(self, config=None):
        self._set_defaults()
        self.config = config
        if config is not None:
            self._load_params(config)

    @abstractmethod
    def _load_params(self, config):

        """Load parameters using config file."""

        pass

    @abstractmethod
    def _set_defaults(self):

        """Default values for your class, if None is passed as config.

        Should initialize the same parameters as in ``_load_params`` method.
        """

        pass
