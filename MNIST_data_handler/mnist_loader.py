import gzip
import struct
import numpy as np
import os

class MNISTLoader:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self._X_train = None
        self._Y_train = None
        self._X_test = None
        self._Y_test = None

    def _load_images(self, image_path):
        with gzip.open(image_path, 'rb') as f:
            magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
            assert magic == 2051, f"Invalid magic number: {magic}"
            images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows * cols)
            return images.astype(np.float32) / 255.0

    def _load_labels(self, label_path):
        with gzip.open(label_path, 'rb') as f:
            magic, num_labels = struct.unpack(">II", f.read(8))
            assert magic == 2049, f"Invalid magic number: {magic}"
            labels = np.frombuffer(f.read(), dtype=np.uint8)
            return labels

    def load_mnist_data(self):
        if self._X_train is None:
            # Load training data
            train_images_path = os.path.join(self.data_dir, 'train-images-idx3-ubyte.gz')
            train_labels_path = os.path.join(self.data_dir, 'train-labels-idx1-ubyte.gz')
            self._X_train = self._load_images(train_images_path)
            self._Y_train = self._load_labels(train_labels_path)

            # Load test data
            test_images_path = os.path.join(self.data_dir, 't10k-images-idx3-ubyte.gz')
            test_labels_path = os.path.join(self.data_dir, 't10k-labels-idx1-ubyte.gz')
            self._X_test = self._load_images(test_images_path)
            self._Y_test = self._load_labels(test_labels_path)

        return self._X_train, self._Y_train, self._X_test, self._Y_test

    def get_training_and_test_data(self, augment=False):
        return self.load_mnist_data()  # Note: augmentation not implemented in this version 