import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras import backend as K

from .patch_resnet_model import PatchModelBuilder
from ..prepare import Preparation
from .base import ModelBuilderBase

if K.image_data_format() == 'channels_last':
    ROW_AXIS = 1
    COL_AXIS = 2
    CHANNEL_AXIS = 3
else:
    CHANNEL_AXIS = 1
    ROW_AXIS = 2
    COL_AXIS = 3


# Helper to build a conv -> BN -> relu block
def _conv_bn_relu(nb_filter, nb_row, nb_col, strides=(1, 1),
                  weight_decay=.0001, dropout=.0, last_block=False):
    def f(input):
        conv = Conv2D(filters=nb_filter, kernel_size=(nb_row, nb_col),
                      strides=strides, kernel_initializer="he_normal",
                      padding="same", kernel_regularizer=l2(weight_decay))(input)
        norm = BatchNormalization(axis=CHANNEL_AXIS)(conv)
        if last_block:
            return norm
        else:
            relu = Activation("relu")(norm)
            return Dropout(dropout)(relu)

    return f


# Helper to build a BN -> relu -> conv block
# This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
def _bn_relu_conv(nb_filter, nb_row, nb_col, strides=(1, 1),
                  weight_decay=.0001, dropout=.0):
    def f(input):
        norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
        activation = Activation("relu")(norm)
        activation = Dropout(dropout)(activation)
        return Conv2D(filters=nb_filter, kernel_size=(nb_row, nb_col),
                      strides=strides, kernel_initializer="he_normal",
                      padding="same",
                      kernel_regularizer=l2(weight_decay))(activation)

    return f


# Adds a shortcut between input and residual block and merges them with "sum"
def _shortcut(input, residual, weight_decay=.0001, dropout=.0, identity=True,
              strides=(1, 1), with_bn=False, org=False):
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    # !!! The dropout argument is just a place holder.
    # !!! It shall not be applied to identity mapping.
    # stride_width = input._keras_shape[ROW_AXIS] // residual._keras_shape[ROW_AXIS]
    # stride_height = input._keras_shape[COL_AXIS] // residual._keras_shape[COL_AXIS]
    # equal_channels = residual._keras_shape[CHANNEL_AXIS] == input._keras_shape[CHANNEL_AXIS]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    # if stride_width > 1 or stride_height > 1 or not equal_channels:
    if not identity:
        shortcut = Conv2D(filters=residual.shape[CHANNEL_AXIS],
                          kernel_size=(1, 1), strides=strides,
                          kernel_initializer="he_normal", padding="valid",
                          kernel_regularizer=l2(weight_decay))(input)
        if with_bn:
            shortcut = BatchNormalization(axis=CHANNEL_AXIS)(shortcut)

    addition = add([shortcut, residual])
    if not org:
        return addition
    else:
        relu = Activation("relu")(addition)
        return Dropout(dropout)(relu)


# Builds a residual block with repeating bottleneck blocks.
def _residual_block(block_function, nb_filters, repetitions,
                    is_first_layer=False, shortcut_with_bn=False,
                    bottleneck_enlarge_factor=4, **kw_args):
    def f(input):
        for i in range(repetitions):
            init_strides = (1, 1)
            identity = True
            if i == 0 and not is_first_layer:
                init_strides = (2, 2)
            if i == 0:
                identity = False
            input = block_function(nb_filters=nb_filters,
                                   init_strides=init_strides,
                                   identity=identity,
                                   shortcut_with_bn=shortcut_with_bn,
                                   enlarge_factor=bottleneck_enlarge_factor,
                                   **kw_args)(input)
        return input

    return f


# Basic 3 X 3 convolution blocks.
# Use for resnet with layers <= 34
# Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
def basic_block(nb_filters, init_strides=(1, 1), identity=True,
                shortcut_with_bn=False, enlarge_factor=None, **kw_args):
    def f(input):
        conv1 = _bn_relu_conv(nb_filters, 3, 3, strides=init_strides, **kw_args)(input)
        residual = _bn_relu_conv(nb_filters, 3, 3, **kw_args)(conv1)
        return _shortcut(input, residual, identity=identity,
                         strides=init_strides,
                         with_bn=shortcut_with_bn, **kw_args)

    return f


def basic_block_org(nb_filters, init_strides=(1, 1), identity=True,
                    shortcut_with_bn=False, enlarge_factor=None, **kw_args):
    def f(input):
        conv1 = _conv_bn_relu(nb_filters, 3, 3, strides=init_strides, **kw_args)(input)
        residual = _conv_bn_relu(nb_filters, 3, 3, last_block=True, **kw_args)(conv1)
        return _shortcut(input, residual, identity=identity,
                         strides=init_strides,
                         with_bn=shortcut_with_bn, org=True, **kw_args)

    return f


# Bottleneck architecture for > 34 layer resnet.
# Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
# Returns a final conv layer of nb_filters * 4
def bottleneck(nb_filters, init_strides=(1, 1), identity=True,
               shortcut_with_bn=False, enlarge_factor=4, **kw_args):
    def f(input):
        conv_1_1 = _bn_relu_conv(nb_filters, 1, 1, strides=init_strides, **kw_args)(input)
        conv_3_3 = _bn_relu_conv(nb_filters, 3, 3, **kw_args)(conv_1_1)
        residual = _bn_relu_conv(nb_filters * enlarge_factor, 1, 1, **kw_args)(conv_3_3)
        return _shortcut(input, residual, identity=identity,
                         strides=init_strides,
                         with_bn=shortcut_with_bn, **kw_args)

    return f


def bottleneck_org(nb_filters, init_strides=(1, 1), identity=True,
                   shortcut_with_bn=False, enlarge_factor=4, **kw_args):
    def f(input):
        conv_1_1 = _conv_bn_relu(nb_filters, 1, 1, strides=init_strides, **kw_args)(input)
        conv_3_3 = _conv_bn_relu(nb_filters, 3, 3, **kw_args)(conv_1_1)
        residual = _conv_bn_relu(nb_filters * enlarge_factor, 1, 1,
                                 last_block=True, **kw_args)(conv_3_3)
        return _shortcut(input, residual, identity=identity,
                         strides=init_strides,
                         with_bn=shortcut_with_bn, org=True, **kw_args)

    return f


def add_top_layers(model, image_size, activation,
                   patch_net='resnet50', block_type='resnet',
                   depths=[512, 512], repetitions=[1, 1],
                   block_fn=bottleneck_org, nb_class=2,
                   shortcut_with_bn=True, bottleneck_enlarge_factor=4,
                   dropout=.0, weight_decay=.0001,
                   add_heatmap=False, avg_pool_size=(7, 7), return_heatmap=False,
                   add_conv=True, add_shortcut=False,
                   hm_strides=(1, 1), hm_pool_size=(5, 5),
                   fc_init_units=64, fc_layers=2):
    def add_residual_blocks(block):
        for depth, repetition in zip(depths, repetitions):
            block = _residual_block(
                block_fn, depth, repetition,
                dropout=dropout, weight_decay=weight_decay,
                shortcut_with_bn=shortcut_with_bn,
                bottleneck_enlarge_factor=bottleneck_enlarge_factor)(block)
        pool = GlobalAveragePooling2D()(block)
        dropped = Dropout(dropout)(pool)
        return dropped

    last_kept_layer = model.layers[-5]

    block = last_kept_layer.output
    channels = 3
    image_input = Input(shape=(image_size[0], image_size[1], channels))
    x = _conv_bn_relu(3, 3, 2, strides=(1, 1), weight_decay=.0001, dropout=.0, last_block=False)(image_input)
    x = _conv_bn_relu(3, 3, 2, strides=(1, 1), weight_decay=.0001, dropout=.0, last_block=False)(x)
    x = AveragePooling2D(pool_size=(4, 3))(x)

    model0 = Model(inputs=model.inputs, outputs=block)
    block = model0(x)
    block = add_residual_blocks(block)
    dense = Dense(nb_class, kernel_initializer="he_normal",
                  activation=activation,
                  kernel_regularizer=l2(weight_decay))(block)
    model_addtop = Model(inputs=image_input, outputs=dense)

    return model_addtop


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
    def __init__(self, config, dropout=True, lr=1e-5, dropout_rate=.3, phase=None) -> None:
        super().__init__(config)
        self. class_names = self.config.general_info.classes
        self.input_shape = (self.config.general_info.image_height_whole,
                            self.config.general_info.image_width_whole,
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
        self.patch_model_weight = self.config.general_info.patch_model_weight

    def get_model(self, hp=None) -> tf.keras.Model:
        """Generates the model for training, and returns the compiled model.

        Returns:
            A compiled ``tensorflow.keras`` model.
        """
        patch_model_cls = PatchModelBuilder(self.config)
        patch_model = patch_model_cls.get_model()
        patch_model.load_weights(self.patch_model_weight)
        image_model = add_top_layers(patch_model,
                                     self.activation,
                                     nb_class=self.classes,
                                     )
        if self.phase == 'train':
            prepare = Preparation()
            metrics_all = prepare.metrics_define(len(self.class_names))
            optimizer = Adam(learning_rate=self.lr)
            image_model.compile(optimizer=optimizer, loss=self.loss, metrics=metrics_all)
        elif self.phase == 'evaluation':
            image_model.load_weights(self.config.general_info.best_weights_path)
        return image_model

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