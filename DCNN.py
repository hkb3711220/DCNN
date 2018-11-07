from keras.models import Model
from keras.layers import Conv2D, MaxPool2D, AvgPool2D, Input, Concatenate, GlobalAvgPool2D, Dense
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.backend import argmax
from keras.utils.np_utils import to_categorical
from utils import CIFAR_10

#BASED ON Densenet-121(K=32)
#Model:https://www.slideshare.net/harmonylab/densely-connected-convolutional-networks
#Essay:https://arxiv.org/pdf/1608.06993.pdf

class DCNN(object):

    def __init__(self):
        self.batch_size = 32
        self.IMG_SIZE = 32
        self.NUM_CLASSES = 10
        self.N_epoch = 10
        self.K = 32 #Growth Rate

    def main(self, inputs):

        self.inputs = inputs

        net = Conv2D(filters=16, kernel_size=(7,7), strides=(2,2), padding='same', activation='relu')(self.inputs)
        net = BatchNormalization()(net)
        net = MaxPool2D(pool_size=(3,3), strides=(2,2), padding='same')(net)

        net, filters  = self._dense_block(net, 6)

        #The transition layers used in our experiments consist of
        #a batch normal-ization layer and an 1×1 convolutional layer followed by a 2×2 average pooling layer

        net = BatchNormalization()(net)
        net = Conv2D(filters=filters,kernel_size=(1,1), strides=(1,1), padding='same')(net)
        net = AvgPool2D(pool_size=(2,2), strides=(2,2), padding='same')(net)

        net, filters = self._dense_block(net, 12)

        net = BatchNormalization()(net)
        net = Conv2D(filters=filters, kernel_size=(1,1), strides=(1,1), padding='same')(net)
        net = AvgPool2D(pool_size=(2,2), strides=(2,2), padding='same')(net)


        net, filters = self._dense_block(net, 24)

        net = BatchNormalization()(net)
        net = Conv2D(filters=filters, kernel_size=(1,1), strides=(1,1), padding='same')(net)
        net = AvgPool2D(pool_size=(2,2), strides=(2,2), padding='same')(net)

        net, filters = self._dense_block(net, 16)
        print(net)

        net = GlobalAvgPool2D(data_format='channels_last')(net)
        net = Dense(self.NUM_CLASSES, activation='softmax')(net)
        print(net)

        return net

    def train(self):

        images = Input(shape=(self.IMG_SIZE, self.IMG_SIZE, 3), dtype='float32', name='images')
        logits = self.main(images)
        model = Model(inputs=images, outputs=logits)
        model.compile(optimizer='Adam',loss='categorical_crossentropy', metrics=['accuracy'])
        #ategorical_crossentropyを使う場合，目的値はカテゴリカルにしなければいけません．

        dataset = CIFAR_10()
        x_train, y_train, x_test, y_test = dataset.train_data, dataset.train_labels, dataset.test_data, dataset.test_labels
        history = model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.N_epoch, verbose=1,
                            validation_data=(x_test, y_test))

        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

    def _dense_block(self, inputs, num_layers):

        inputs_pool = []
        inputs_pool.append(inputs)
        num_input = inputs.get_shape().as_list()[3]

        for i in range(num_layers):
            if i == 0:
                #First layer
                net = BatchNormalization()(inputs)
            else:
                #Second layer -> Last Layer
                net_input = Concatenate(axis=3)(inputs_pool)
                net = BatchNormalization()(net_input)

            net = Activation('relu')(net)
            #In our experiments, we let each 1×1 convolution produce 4k feature-maps.
            net = Conv2D(filters=self.K*4, kernel_size=(1,1), strides=(1,1), padding='same')(net)
            net = BatchNormalization()(net)
            net = Activation('relu')(net)
            net = Conv2D(filters=self.K*i+num_input, kernel_size=(3,3), strides=(1,1), padding='same')(net)
            inputs_pool.append(net)

        filters = net.get_shape().as_list()[3]

        return net, filters


if __name__ == '__main__':
    DCNN().train()
