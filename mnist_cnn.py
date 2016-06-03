"""
Adventures in Deep Learning

Convolutional Neural Network for MNIST dataset.
The architecture of the network is taken from
https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py

Goker Erdogan
https://github.com/gokererdogan
"""

import numpy as np
import keras.models as kmodel
import keras.layers.convolutional as kconv
import keras.layers.core as klcore
from keras.utils.np_utils import to_categorical

from common import load_mnist

(x_train, y_train), (x_val, y_val), (x_test, y_test) = load_mnist()
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)

# convert to 2D images
x_train = np.reshape(x_train, (-1, 1, 28, 28))
x_val = np.reshape(x_val, (-1, 1, 28, 28))
x_test = np.reshape(x_test, (-1, 1, 28, 28))

model = kmodel.Sequential()

model.add(kconv.Convolution2D(nb_filter=32, nb_row=5, nb_col=5, input_shape=(1, 28, 28), border_mode='valid'))
model.add(klcore.Activation('tanh'))
model.add(kconv.Convolution2D(nb_filter=32, nb_row=5, nb_col=5))
model.add(klcore.Activation('tanh'))
model.add(kconv.MaxPooling2D(pool_size=(2, 2)))
model.add(klcore.Flatten())
model.add(klcore.Dense(output_dim=128))
model.add(klcore.Activation('tanh'))
model.add(klcore.Dense(output_dim=10))
model.add(klcore.Activation('softmax'))

model.compile(optimizer='adadelta', loss='categorical_crossentropy')

model.fit(x=x_train, y=y_train, nb_epoch=1, batch_size=128, validation_data=(x_val, y_val))

# evaluate on test set
pred_y = model.predict_classes(x=x_test, verbose=1)

# calculate accuracy
acc = np.mean(pred_y == y_test)
print("Accuracy: {0:f}".format(acc))


