"""
Adventures in Deep Learning

A simple example of long-short term memory model.

Goker Erdogan
https://github.com/gokererdogan
"""
import numpy as np
import theano
from theano import tensor as T

from collections import OrderedDict


class LSTMCell(object):
    def __init__(self, input, input_dimension, memory_size):
        self.input = input
        self.input_dimension = input_dimension
        self.memory_size = memory_size

        # input gate
        self.wi = theano.shared(value=init_array((input_dimension, 1)), name='wi')
        self.bi = theano.shared(value=init_array(1), name='bi')
        self.wih = theano.shared(value=init_array((memory_size, 1)), name='wih')

        # write cell content
        self.wc = theano.shared(value=init_array((input_dimension, memory_size)), name='wc')
        self.bc = theano.shared(value=init_array(memory_size), name='bc')
        self.wch = theano.shared(value=init_array((memory_size, memory_size)), name='wch')

        # forget gate
        self.wf = theano.shared(value=init_array((input_dim, 1)), name='wf')
        self.bf = theano.shared(value=init_array(1), name='bf')
        self.wfh = theano.shared(value=init_array((memory_size, 1)), name='wfh')

        # output gate
        self.wo = theano.shared(value=init_array((input_dim, 1)), name='wo')
        self.bo = theano.shared(value=init_array(1), name='bo')
        self.woh = theano.shared(value=init_array((memory_size, 1)), name='woh')

        self.params = [self.wi, self.bi, self.wih, self.wc, self.bc, self.wch, self.wf, self.bf, self.wfh,
                       self.wo, self.bo, self.woh]

        # initial memory content and hidden state
        self.c0 = theano.shared(value=np.zeros(memory_size, dtype=theano.config.floatX), name='c0')
        self.h0 = theano.shared(value=np.zeros(memory_size, dtype=theano.config.floatX), name='h0')

    def forward_pass(self):
        def recurrence(x_t, h_tm1, c_tm1):
            i = T.nnet.sigmoid(T.dot(x_t, self.wi) + T.dot(h_tm1, self.wih) + self.bi)  # input gate
            c_proposed = T.tanh(T.dot(x_t, self.wc) + T.dot(h_tm1, self.wch) + self.bc)  # proposed memory cell content
            f = T.nnet.sigmoid(T.dot(x_t, self.wf) + T.dot(h_tm1, self.wfh) + self.bf)  # forget gate
            c_t = (T.tile(i, self.memory_size) * c_proposed) + (T.tile(f, self.memory_size) * c_tm1)  # new memory cell content
            o = T.nnet.sigmoid(T.dot(x_t, self.wo) + T.dot(h_tm1, self.woh) + self.bo)  # output gate
            h_t = T.tile(o, self.memory_size) * T.tanh(c_t)
            return [h_t, c_t]

        [h, c], _ = theano.scan(fn=recurrence, sequences=self.input,
                                outputs_info=[self.h0, self.c0], n_steps=self.input.shape[0])

        return h, c


def generate_input_output_pair(dim):
    length = np.random.randint(5, 20)
    # length = 1
    input = np.random.randint(0, 2, (length, dim)).astype(theano.config.floatX)
    output = input[0]
    return input, output


def init_array(size):
    return 0.2 * np.random.uniform(-1.0, 1.0, size).astype(theano.config.floatX)

# input
input_dim = 3
output_dim = 3
x = T.matrix('x')
y = T.vector('y')

# PARAMETERS
memory_cell_size = 10
memory_cell_count = 2

cell1 = LSTMCell(x, input_dim, memory_cell_size)
cell2 = LSTMCell(x, input_dim, memory_cell_size)

# hidden to output
wy = theano.shared(value=init_array((memory_cell_size*memory_cell_count, output_dim)), name='wy')
by = theano.shared(value=init_array(output_dim), name='by')

h1, c1 = cell1.forward_pass()
h2, c2 = cell2.forward_pass()
prediction = T.nnet.sigmoid(T.sum(T.dot(T.concatenate([h1, h2], axis=1), wy) + by, axis=0))

nll = T.mean(T.nnet.binary_crossentropy(prediction, y))

params = cell1.params + cell2.params + [wy, by]

# training
lr = 0.005

dparams = T.grad(nll, params)
updates = OrderedDict({p: (p - lr*dp) for p, dp in zip(params, dparams)})

train = theano.function(inputs=[x, y], outputs=nll, updates=updates)
test = theano.function(inputs=[x, y], outputs=nll)
predict = theano.function(inputs=[x], outputs=prediction)

# number of training epochs, i.e., passes over training set.
epoch_count = 10

# report log_likelihood bound on training after this many training samples
report_interval = 1000

# number of training samples
train_n = 5000

# arrays for keeping track of log ll bounds on training and validation sets
train_ll = []
test_ll = []

# TRAINING LOOP
for e in xrange(epoch_count):
    ll = 0.0
    for i in xrange(0, train_n):
        xi, yi = generate_input_output_pair(input_dim)
        cll = train(xi, yi)  # one step of gradient descent
        ll += cll
        # report log ll bound
        if (i+1) % report_interval == 0:
            ll /= report_interval
            print "|Epoch {0:d}|\tTrain ll: {1:f}".format(e+1, ll)
            train_ll.append(ll)
            ll = 0.0

print("Training over.")
