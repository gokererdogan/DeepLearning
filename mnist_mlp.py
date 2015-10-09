"""
Adventures in Deep Learning

Multilayer Perceptron for MNIST Handwritten Digit Recognition

Goker Erdogan
https://github.com/gokererdogan
"""

import gmllib.dataset as dataset
import gmllib.helpers as hlp
from keras.models import Sequential
from keras.layers.core import Dense, Activation

# load data
ds_name = 'mnist'
ds_path = '../gmllib/datasets/mnist'
ds = dataset.DataSet.load_from_path(name=ds_name, folder=ds_path)


# build model
model = Sequential()
model.add(Dense(input_dim=784, output_dim=400))
model.add(Activation("sigmoid"))
model.add(Dense(input_dim=400, output_dim=10))
model.add(Activation("softmax"))

model.compile(optimizer="sgd", loss="categorical_crossentropy")

# fit model
model.fit(X=ds.train.x, y=ds.train.y, batch_size=128, nb_epoch=5, verbose=1,
          validation_data=(ds.validation.x, ds.validation.y))


# evaluate on test set
pred_y = model.predict_classes(X=ds.test.x, verbose=1)

# calculate accuracy
acc = np.mean(pred_y == hlp.convert_1ofK_to_ordinal(ds.test.y))
print("Accuracy: {0:f}".format(acc))
