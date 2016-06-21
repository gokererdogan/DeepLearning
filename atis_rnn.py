"""
Adventures in Deep Learning

Recurrent Neural Network for ATIS Slot Filling task.
This is an implementation of the model discussed here http://www.deeplearning.net/tutorial/rnnslu.html

Goker Erdogan
https://github.com/gokererdogan
"""
import cPickle as pkl

import numpy as np
import theano
from theano import tensor as T


def init_array(size):
    return 0.2 * np.random.uniform(-1.0, 1.0, size).astype(theano.config.floatX)

# read dataset
train, test, dicts = pkl.load(open('datasets/atis.pkl'))

train_x, _, train_y = train
test_x, _, test_y = test

idx2words = {i: w for w, i in dicts['words2idx'].iteritems()}
idx2labels = {i: w for w, i in dicts['labels2idx'].iteritems()}

# word embeddings
word_count = len(idx2words)
embed_dim = 50

embeddings = theano.shared(value=init_array((word_count+1, embed_dim)), name='embeddings')

idxs = T.ivector('idxs')
x = embeddings[idxs]
y = T.ivector('y')

getx = theano.function([idxs], x)

# parameters
hidden_count = 100
wx = theano.shared(value=init_array((embed_dim, hidden_count)), name='wx')

h0 = theano.shared(value=np.zeros(hidden_count, dtype=theano.config.floatX), name='h0')
wh = theano.shared(value=init_array((hidden_count, hidden_count)), name='wh')

class_count = len(idx2labels)
w = theano.shared(value=init_array((hidden_count, class_count)), name='w')
b = theano.shared(value=init_array(class_count), name='b')

# calculate output
def recurrence(x_t, h_tm1):
    h_t = T.nnet.sigmoid(T.dot(x_t, wx) + T.dot(h_tm1, wh))
    s_t = T.nnet.softmax(T.dot(h_t, w) + b)
    return [h_t, s_t]

[h, s], _ = theano.scan(fn=recurrence, sequences=x, outputs_info=[h0, None], n_steps=x.shape[0])

p_yx = s[:, 0, :]
y_pred = T.argmax(p_yx, axis=1)

# training
lr = 0.01
nll = -T.mean(T.log(p_yx[T.arange(x.shape[0]), y]))

de, dwx, dh0, dwh, dw, db = T.grad(nll, [embeddings, wx, h0, wh, w, b])

updates = ((embeddings, embeddings - lr*de), (wx, wx - lr*dwx), (h0, h0 - lr*dh0),
           (wh, wh - lr*dwh), (w, w - lr*dw), (b, b - lr*db))

train = theano.function(inputs=[idxs, y], outputs=nll, updates=updates)
test = theano.function(inputs=[idxs, y], outputs=nll)
classify = theano.function(inputs=[idxs], outputs=y_pred)

normalize = theano.function(inputs=[], updates={embeddings:
                                                embeddings / T.sqrt((embeddings**2).sum(axis=1)).dimshuffle(0, 'x')})


# number of training epochs, i.e., passes over training set.
epoch_count = 50

# report log_likelihood bound on training after this many training samples
report_interval = 1000


# arrays for keeping track of log ll bounds on training and validation sets
train_ll = []
test_ll = []

# TRAINING LOOP
for e in xrange(epoch_count):
    ll = 0.0
    for i in xrange(0, len(train_x)):
        cll = train(train_x[i], train_y[i])  # one step of gradient descent
        normalize()
        ll += cll
        # report log ll bound
        if (i+1) % report_interval == 0:
            ll /= report_interval
            print "|Epoch {0:d}|\tTrain ll: {1:f}".format(e+1, ll)
            train_ll.append(ll)
            ll = 0.0

    # epoch over, shuffle training data
    rp = np.random.permutation(len(train_x))
    train_x = [train_x[i] for i in rp]
    train_y = [train_y[i] for i in rp]

    # calculate log ll on test set
    ll = 0.0
    for i in xrange(0, len(test_x)):
        cll = test(test_x[i], test_y[i])
        ll += cll

    ll /= len(test_x)
    print "|Epoch {0:d}|\tTest ll: {1:f}".format(e+1, ll)
    test_ll.append(ll)

print("Training over.")

