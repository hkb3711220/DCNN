import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical

os.chdir(os.path.dirname(__file__))

class CIFAR_10(object):

    def __init__(self):

        self.PATH = './Dataset'
        files = os.listdir(self.PATH)
        self.label_names = self._unpickle(files[0])[b"label_names"]

        self.train_data, self.train_labels = self._load_data(files[1:6])
        self.test_data, self.test_labels = self._load_data(files[6])

    def _unpickle(self, file):

        file_path = os.path.join(self.PATH, file)
        with open(file_path, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')

        return dict

    def _load_data(self, file_names):

        data = []
        labels = []

        if file_names == 'test_batch':
            pictures = self._unpickle(file_names)[b"data"]
            labels_set = self._unpickle(file_names)[b"labels"]

            for picture in pictures:
                data.append(self._load_pictures(picture))
            for label in labels_set:
                labels.append(label)

        else:
            for file_name in file_names:
                pictures = self._unpickle(file_name)[b"data"]
                labels_set = self._unpickle(file_name)[b"labels"]

                for picture in pictures:
                    data.append(self._load_pictures(picture))
                for label in labels_set:
                    labels.append(label)

        labels = self._one_hot_encoding(labels, self.label_names)

        return np.asarray(data), labels

    def _load_pictures(self, data):

        img = data / 255 #画像正規化
        img = img.reshape([3, 32, 32]).transpose([1, 2, 0])

        return img

    def _one_hot_encoding(self, data, label_names):

        NUM_CLASSES = len(label_names)
        one_hot_data = to_categorical(data, NUM_CLASSES)

        return one_hot_data
        
