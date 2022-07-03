"""
Data Generator
"""

import cv2
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from tensorflow.keras.utils import Sequence
from .augmentation import Augmentation
import typing


class ImageSequenceMultiView(Sequence):

    def __init__(self,
                 dataset_csv_file: typing.Optional[pd.DataFrame],
                 cc_names: str,
                 mlo_names: str,
                 class_names: str,
                 source_image_dir: str,
                 batch_size: int = 16,
                 target_size_h: int = 224,
                 target_size_w: int = 224,
                 synthesize: bool = False,
                 augmenter: typing.Optional[Augmentation] = None,
                 verbose: bool = False,
                 steps: int = None,
                 shuffle_on_epoch_end: bool = True,
                 random_state: int = 1
                 ):
        """
        This class is used to create a multi-vies tf-data-generator
        Args:
            dataset_csv_file: the train/val/test dataframe
            cc_names: the column name of the dataframe that contains paths to cc images
            mlo_names: the column name of the dataframe that contains paths to mlo images
            class_names: the column names of the dataframe that contain label of the images
            source_image_dir: path to the source directory of the images
            batch_size:
            target_size_h:
            target_size_w:
            synthesize: boolean indicating whether to synthesize 3-channel images or not
            augmenter: either the augmentation class on None
            verbose: boolean indicating whether to visualize samples of data or not
            steps: data-generator steps
            shuffle_on_epoch_end:
            random_state:
        """
        self.dataset_df = dataset_csv_file
        self.source_image_dir = source_image_dir
        self.batch_size = batch_size
        self.H = target_size_h
        self.W = target_size_w
        self.synthesize_3_channel = synthesize
        self.augmenter = augmenter
        self.verbose = verbose
        self.shuffle = shuffle_on_epoch_end
        self.random_state = random_state
        self.class_names = class_names
        self.cc_names = cc_names
        self.mlo_names = mlo_names
        self.cc_path = None
        self.mlo_path = None
        self.y = None
        self.prepare_dataset()

        if steps is None:
            self.steps = int(np.ceil(len(self.cc_path) / float(self.batch_size)))
        else:
            self.steps = int(steps)

        if self.verbose:
            self.visualize()

    def __bool__(self):
        return True

    def __len__(self):
        return self.steps

    @staticmethod
    def synthesize(img):

        clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(5, 5))
        cl1 = clahe.apply(np.array(img, dtype=np.uint8))
        cl1 = (cl1 - cl1.min()) / (cl1.max() - cl1.min())

        clahe = cv2.createCLAHE(clipLimit=3)
        cl2 = clahe.apply(np.array(img, dtype=np.uint8))
        cl2 = (cl2 - cl2.min()) / (cl2.max() - cl2.min())

        img = (img - img.min()) / (img.max() - img.min())
        synthesized = cv2.merge((img, cl1, cl2))
        return synthesized

    def __getitem__(self, idx):
        batch_cc_path = self.cc_path[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_mlo_path = self.mlo_path[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_cc = np.asarray([self.load_image(cc_path) for cc_path in batch_cc_path])

        batch_mlo = np.asarray([self.load_image(mlo_path) for mlo_path in batch_mlo_path])

        batch_x = [batch_cc, batch_mlo]
        batch_x = self.transform_batch_images(batch_x)

        return batch_x, batch_y

    def load_image(self, image_file):
        image_path = os.path.join(self.source_image_dir + image_file)
        image = cv2.imread(image_path, 0)
        image = cv2.resize(image, (self.W, self.H))
        if self.synthesize_3_channel:
            image_array = self.synthesize(image)
        else:
            clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(5, 5))
            image = clahe.apply(np.array(image, dtype=np.uint8))
            image = (image - image.min()) / (image.max() - image.min())
            image = np.expand_dims(image, axis=-1)
            image_array = np.concatenate([image, image, image], axis=-1)
        return image_array

    def transform_batch_images(self, batch_x):
        if self.augmenter is not None:
            aug = self.augmenter
            data = aug.batch_augmentation(batch_x)
            batch_cc = data[0]
            batch_mlo = data[1]
        else:
            batch_cc = batch_x[0]
            batch_mlo = batch_x[1]
        return [batch_cc, batch_mlo]

    def get_y_true(self):
        """
        Use this function to get y_true for predict_generator
        In order to get correct y, you have to set shuffle_on_epoch_end=False.

        """
        if self.shuffle:
            raise ValueError("""
            You're trying run get_y_true() when generator option 'shuffle_on_epoch_end' is True.
            """)
        return self.y[:self.steps * self.batch_size, :]

    def get_x_true(self):
        """
        Use this function to get y_true for predict_generator
        In order to get correct y, you have to set shuffle_on_epoch_end=False.

        """
        if self.shuffle:
            raise ValueError("""
            You're trying run get_y_true() when generator option 'shuffle_on_epoch_end' is True.
            """)
        return self.cc_path[:self.steps * self.batch_size], self.mlo_path[:self.steps * self.batch_size]

    def prepare_dataset(self):
        df = self.dataset_df.sample(frac=1., random_state=self.random_state)
        self.cc_path, self.mlo_path, self.y = df[self.cc_names].values, df[self.mlo_names].values, df[
            self.class_names].values.astype(np.float32)

    def on_epoch_end(self):
        if self.shuffle:
            self.random_state += 1
            self.prepare_dataset()

    def visualize(self):
        [cc, mlo], label = self.__getitem__(10)

        for i in range(len(2)):
            plt.figure(figsize=(6, 3))
            plt.subplot(1, 2, 1)
            plt.imshow(cc[i], cmap='gray')
            plt.axis('off')
            plt.title('CC')
            plt.subplot(1, 2, 2)
            plt.imshow(mlo[i], cmap='gray')
            plt.axis('off')
            plt.title('MLO')
            plt.show()


class ImageSequenceSingleView(Sequence):

    def __init__(self,
                 dataset_csv_file: typing.Optional[pd.DataFrame],
                 x_names: str,
                 class_names: str,
                 source_image_dir: str,
                 batch_size: int = 16,
                 target_size_h: int = 224,
                 target_size_w: int = 224,
                 synthesize: bool = False,
                 augmenter: typing.Optional[Augmentation] = None,
                 verbose: bool = False,
                 steps: int = None,
                 shuffle_on_epoch_end: bool = True,
                 random_state: int = 1
                 ):
        """
        This class is used to create a single-view tf-data-generator
        Args:
            dataset_csv_file: the train/val/test dataframe
            x_names: the column name of the dataframe that contains paths to images
            class_names: the column names of the dataframe that contain label of the images
            source_image_dir: path to the source directory of the images
            batch_size:
            target_size_h:
            target_size_w:
            synthesize: boolean indicating whether to synthesize 3-channel images or not
            augmenter: either the augmentation class on None
            verbose: boolean indicating whether to visualize samples of data or not
            steps: data-generator steps
            shuffle_on_epoch_end:
            random_state:
        """
        self.dataset_df = dataset_csv_file
        self.source_image_dir = source_image_dir
        self.batch_size = batch_size
        self.H = target_size_h
        self.W = target_size_w
        self.synthesize_3_channel = synthesize
        self.augmenter = augmenter
        self.verbose = verbose
        self.shuffle = shuffle_on_epoch_end
        self.random_state = random_state
        self.class_names = class_names
        self.x_names = x_names
        self.x_path = None
        self.y = None
        self.prepare_dataset()

        if steps is None:
            self.steps = int(np.ceil(len(self.cc_path) / float(self.batch_size)))
        else:
            self.steps = int(steps)

        if self.verbose:
            self.visualize()

    def __bool__(self):
        return True

    def __len__(self):
        return self.steps

    @staticmethod
    def synthesize(img):

        clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(5, 5))
        cl1 = clahe.apply(np.array(img, dtype=np.uint8))
        cl1 = (cl1 - cl1.min()) / (cl1.max() - cl1.min())

        clahe = cv2.createCLAHE(clipLimit=3)
        cl2 = clahe.apply(np.array(img, dtype=np.uint8))
        cl2 = (cl2 - cl2.min()) / (cl2.max() - cl2.min())

        img = (img - img.min()) / (img.max() - img.min())
        synthesized = cv2.merge((img, cl1, cl2))
        return synthesized

    def __getitem__(self, idx):
        batch_x_path = self.x_path[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_x = np.asarray([self.load_image(x_path) for x_path in batch_x_path])

        batch_x_new = [batch_x, batch_x]
        batch_x = self.transform_batch_images(batch_x_new)

        return batch_x[0], batch_y

    def load_image(self, image_file):
        image_path = os.path.join(self.source_image_dir + image_file)
        image = cv2.imread(image_path, 0)
        image = cv2.resize(image, (self.W, self.H))
        if self.synthesize_3_channel:
            image_array = self.synthesize(image)
        else:
            clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(5, 5))
            image = clahe.apply(np.array(image, dtype=np.uint8))
            image = (image - image.min()) / (image.max() - image.min())
            image = np.expand_dims(image, axis=-1)
            image_array = np.concatenate([image, image, image], axis=-1)
        return image_array

    def transform_batch_images(self, batch_x):
        if self.augmenter is not None:
            aug = self.augmenter
            data = aug.batch_augmentation(batch_x)
            batch_cc = data[0]
            batch_mlo = data[1]
        else:
            batch_cc = batch_x[0]
            batch_mlo = batch_x[1]
        return [batch_cc, batch_mlo]

    def get_y_true(self):
        """
        Use this function to get y_true for predict_generator
        In order to get correct y, you have to set shuffle_on_epoch_end=False.

        """
        if self.shuffle:
            raise ValueError("""
            You're trying run get_y_true() when generator option 'shuffle_on_epoch_end' is True.
            """)
        return self.y[:self.steps * self.batch_size, :]

    def get_x_true(self):
        """
        Use this function to get y_true for predict_generator
        In order to get correct y, you have to set shuffle_on_epoch_end=False.

        """
        if self.shuffle:
            raise ValueError("""
            You're trying run get_y_true() when generator option 'shuffle_on_epoch_end' is True.
            """)
        return self.x_path[:self.steps * self.batch_size]

    def prepare_dataset(self):
        df = self.dataset_df.sample(frac=1., random_state=self.random_state)
        self.x_path, self.y = df[self.x_names].values, df[self.class_names].values.astype(np.float32)

    def on_epoch_end(self):
        if self.shuffle:
            self.random_state += 1
            self.prepare_dataset()

    def visualize(self):
        img, label = self.__getitem__(10)

        plt.figure(figsize=(6, 3))
        plt.subplot(1, 2, 1)
        plt.imshow(img[0], cmap='gray')
        plt.axis('off')
        plt.title('Sample Image 1')
        plt.subplot(1, 2, 2)
        plt.imshow(img[-1], cmap='gray')
        plt.axis('off')
        plt.title('Sample Image 1')
        plt.show()
