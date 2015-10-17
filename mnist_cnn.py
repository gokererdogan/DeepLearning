"""
Adventures in Deep Learning

Convolutional Neural Network for MNIST dataset.
The architecture of the network is taken from
https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py

Goker Erdogan
https://github.com/gokererdogan
"""

import numpy as np
import gmllib.dataset as dataset
import keras.models as kmodel
import keras.layers.convolutional as kconv
import keras.layers.core as klcore

ds = dataset.DataSet.load_from_path('mnist', '../gmllib/datasets/mnist')

# convert to 2D images
x_train = np.reshape(ds.train.x, (ds.train.N, 1, 28, 28))
x_test = np.reshape(ds.test.x, (ds.test.N, 1, 28, 28))

model = kmodel.Sequential()

model.add(kconv.Convolution2D(nb_filter=32, nb_row=5, nb_col=5, input_shape=(1, 28, 28), border_mode='full'))
model.add(klcore.Activation('tanh'))
model.add(kconv.Convolution2D(nb_filter=32, nb_row=5, nb_col=5))
model.add(klcore.Activation('tanh'))
model.add(kconv.MaxPooling2D(pool_size=(2, 2)))
model.add(klcore.Flatten())
model.add(klcore.Dense(output_dim=128))
model.add(klcore.Activation('tanh'))
model.add(klcore.Dense(output_dim=10))
model.add(klcore.Activation('softmax'))

model.compile(optimizer='adadelta', loss='categorical_crossentropy', class_mode='categorical')

model.fit(X=x_train, y=ds.train.y, nb_epoch=10, validation_split=0.2, show_accuracy=1)

# evaluate on test set
pred_y = model.predict_classes(X=x_test, verbose=1)

# calculate accuracy
acc = np.mean(pred_y == ds.test.y)
print("Accuracy: {0:f}".format(acc))


