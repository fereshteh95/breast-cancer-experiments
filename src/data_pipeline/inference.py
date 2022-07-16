import typing
import pandas as pd

from .data import ImageSequenceMultiView, ImageSequenceSingleView
from .augmentation import Augmentation
from prepare import Preparation
from .base import DataLoaderBase


class DataLoader(DataLoaderBase):
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
        super().__init__(config)
        self.prepare = Preparation()
        self.multi_view = self.config.general_info.multi_view
        self.random_seed = self.config.general_info.random_seed
        self.train_csv = self.config.data_sequence.train_csv_file
        self.validation_csv = self.config.data_sequence.validation_csv_file
        self.test_csv = self.config.data_sequence.test_csv_file
        self.train_df = pd.read_csv(self.train_csv)
        self.validation_df = pd.read_csv(self.validation_csv)
        self.test_df = pd.read_csv(self.test_csv)
        self.image_source_dir_train = self.config.general_info.image_source_dir_train
        self.image_source_dir_val = self.config.general_info.image_source_dir_val
        self.image_source_dir_test = self.config.general_info.image_source_dir_test
        self.class_names = self.config.general_info.classes
        self.x_names = self.config.data_sequence.x_names
        self.cc_names = self.config.data_sequence.cc_names
        self.mlo_names = self.config.data_sequence.mlo_names
        self.target_size_h = self.config.data_sequence.image_height
        self.target_size_w = self.config.data_sequence.image_width
        self.synthesize = self.config.data_sequence.synthesize
        self.shuffle_on_epoch_end = self.config.data_sequence.shuffle_on_epoch_end
        self.augmenter = self.config.data_sequence.augmenter
        self.verbose = self.config.data_sequence.verbose
        self.train_batch_size = self.config.general_info.train_batch_size
        self.valid_batch_size = self.config.general_info.val_batch_size
        self.test_batch_size = self.config.general_info.test_batch_size
        self.patch_training = self.config.general_info.patch_training

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
        train_steps = self.prepare.steps(self.train_df, self.class_names, self.train_batch_size)

        if self.augmenter:
            augmentation = Augmentation
        else:
            augmentation = None

        img_h = self.target_size_h
        img_w = self.target_size_w

        if self.multi_view:
            train_sequence = ImageSequenceMultiView(self.train_df,
                                                    self.cc_names,
                                                    self.mlo_names,
                                                    self.class_names,
                                                    self.image_source_dir_train,
                                                    self.train_batch_size,
                                                    img_h,
                                                    img_w,
                                                    self.synthesize,
                                                    augmentation,
                                                    self.verbose,
                                                    train_steps,
                                                    self.shuffle_on_epoch_end,
                                                    self.random_seed
                                                    )
        else:
            train_sequence = ImageSequenceSingleView(self.train_df,
                                                     self.x_names,
                                                     self.class_names,
                                                     self.image_source_dir_train,
                                                     self.train_batch_size,
                                                     img_h,
                                                     img_w,
                                                     self.synthesize,
                                                     augmentation,
                                                     self.verbose,
                                                     train_steps,
                                                     self.shuffle_on_epoch_end,
                                                     self.random_seed
                                                     )

        return train_sequence, train_steps

    def get_class_weight(self) -> typing.Optional[dict]:
        """Set this if you want to pass ``class_weight`` to ``fit``.

        Returns:
           Optional dictionary mapping class indices (integers) to a weight (float) value.
           used for weighting the loss function (during training only).
           This can be useful to tell the model to "pay more attention" to samples from an under-represented class.

        """
        class_weights = {}

        for i in range(len(self.class_names)):
            class_weights[i] = 1 - len(self.train_df.loc[self.train_df[self.class_names[i]] == 1]) / len(self.train_df)

        return class_weights

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

        val_steps = self.prepare.steps(self.validation_df, self.class_names, self.valid_batch_size)

        augmentation = None

        img_h = self.target_size_h
        img_w = self.target_size_w

        if self.multi_view:
            val_sequence = ImageSequenceMultiView(self.validation_df,
                                                  self.cc_names,
                                                  self.mlo_names,
                                                  self.class_names,
                                                  self.image_source_dir_val,
                                                  self.valid_batch_size,
                                                  img_h,
                                                  img_w,
                                                  self.synthesize,
                                                  augmentation,
                                                  self.verbose,
                                                  val_steps,
                                                  False,
                                                  self.random_seed
                                                  )
        else:
            val_sequence = ImageSequenceSingleView(self.validation_df,
                                                   self.x_names,
                                                   self.class_names,
                                                   self.image_source_dir_val,
                                                   self.valid_batch_size,
                                                   img_h,
                                                   img_w,
                                                   self.synthesize,
                                                   augmentation,
                                                   self.verbose,
                                                   val_steps,
                                                   False,
                                                   self.random_seed
                                                   )
        return val_sequence, val_steps

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

        test_steps = self.prepare.steps(self.test_df, self.class_names, self.test_batch_size)

        augmentation = None

        img_h = self.target_size_h
        img_w = self.target_size_w

        if self.multi_view:
            test_sequence = ImageSequenceMultiView(self.test_df,
                                                   self.cc_names,
                                                   self.mlo_names,
                                                   self.class_names,
                                                   self.image_source_dir_test,
                                                   self.test_batch_size,
                                                   img_h,
                                                   img_w,
                                                   self.synthesize,
                                                   augmentation,
                                                   True,
                                                   test_steps,
                                                   False,
                                                   self.random_seed
                                                   )
        else:
            test_sequence = ImageSequenceSingleView(self.test_df,
                                                    self.x_names,
                                                    self.class_names,
                                                    self.image_source_dir_test,
                                                    self.test_batch_size,
                                                    img_h,
                                                    img_w,
                                                    self.synthesize,
                                                    augmentation,
                                                    True,
                                                    test_steps,
                                                    False,
                                                    self.random_seed
                                                    )
        return test_sequence, test_steps
