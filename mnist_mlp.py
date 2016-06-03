"""
Adventures in Deep Learning

Multilayer Perceptron for MNIST Handwritten Digit Recognition

Goker Erdogan
https://github.com/gokererdogan
"""
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils.np_utils import to_categorical

from common import load_mnist

# load data
(x_train, y_train), (x_val, y_val), (x_test, y_test) = load_mnist()
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)

# build model
model = Sequential()
model.add(Dense(input_dim=784, output_dim=400))
model.add(Activation("sigmoid"))
model.add(Dense(input_dim=400, output_dim=10))
model.add(Activation("softmax"))

model.compile(optimizer="sgd", loss="categorical_crossentropy")

# fit model
model.fit(x=x_train, y=y_train, batch_size=128, nb_epoch=5, verbose=1,
          validation_data=(x_val, y_val))


# evaluate on test set
pred_y = model.predict_classes(x=x_test, verbose=1)

# calculate accuracy
acc = np.mean(pred_y == y_test)
print("Accuracy: {0:f}".format(acc))
