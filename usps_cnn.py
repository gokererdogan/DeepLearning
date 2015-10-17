"""
Adventures in Deep Learning

Convolutional Neural Network for USPS Zipcode dataset
The network here is an implementation of the network
in LeCun et al. (1990) Handwrittten Digit Recognition
with a Back-propogation Network with some minor
differences.

Goker Erdogan
https://github.com/gokererdogan
"""

import numpy as np
import gmllib.dataset as dataset
import gmllib.helpers as hlp
import keras.models as kmodel
import keras.layers.convolutional as kconv
import keras.layers.core as klcore

ds = dataset.DataSet.load_from_path('usps', '../gmllib/datasets/usps')

# convert to 2D images
x_train = np.reshape(ds.train.x, (ds.train.N, 1, 16, 16))
x_test = np.reshape(ds.test.x, (ds.test.N, 1, 16, 16))

model = kmodel.Sequential()

model.add(kconv.Convolution2D(nb_filter=4, nb_row=5, nb_col=5, input_shape=(1, 16, 16), border_mode='full'))
model.add(klcore.Activation('tanh'))
# instead of average pooling, we use max pooling
model.add(kconv.MaxPooling2D(pool_size=(2, 2)))

# the 12 feature maps in this layer are connected in a specific pattern to the below layer, but it is not possible
# do this in keras easily. in fact, I don't know how keras connects the feature maps in one layer to the next.
model.add(kconv.Convolution2D(nb_filter=12, nb_row=5, nb_col=5))
model.add(klcore.Activation('tanh'))
model.add(kconv.MaxPooling2D(pool_size=(2, 2)))

model.add(klcore.Flatten())
model.add(klcore.Dense(output_dim=10))
model.add(klcore.Activation('softmax'))


model.compile(optimizer='sgd', loss='categorical_crossentropy', class_mode='categorical')

model.fit(X=x_train, y=ds.train.y, nb_epoch=50, validation_split=0.2)

# evaluate on test set
pred_y = model.predict_classes(X=x_test, verbose=1)

# calculate accuracy
acc = np.mean(pred_y == hlp.convert_1ofK_to_ordinal(ds.test.y))
print("Accuracy: {0:f}".format(acc))


