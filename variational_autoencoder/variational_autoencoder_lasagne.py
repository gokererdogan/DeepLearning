"""
A Lasagne implementation of the variational autoencoder proposed in
    Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes.
    arXiv:1312.6114

28 May 2016
goker erdogan
https://github.com/gokererdogan
"""
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

import lasagne

from DeepLearning.common import load_mnist


class NormalSamplingLayer(lasagne.layers.MergeLayer):
    """
    This class implements a sampling layer that generates variables from a multivariate normal distribution with
    calculated mean and diagonal covariance matrix.
    This layer is used in the encoder network of the variational autoencoder for sampling the latent variables.

    Parameters:
        incomings (2-tuple): A tuple of lasagne layers. First calculates the mean vector, and second calculates the
            standard deviations for the normal distribution to sample from.
        rng (RandomStreams instance): Random number generator
    """
    def __init__(self, incomings, rng, **kwargs):
        super(NormalSamplingLayer, self).__init__(incomings=incomings, **kwargs)
        self.rng = rng
        if len(incomings) != 2:
            raise ValueError("NormalSamplingLayer requires two inputs.")

    def get_output_for(self, inputs, **kwargs):
        # Sample z from Normal(z|mu, sigma) using the reparameterization trick.
        eps = srng.normal((minibatch_size, latent_dim))  # draw random normal values

        mu = inputs[0]
        sigma = inputs[1]
        z = mu + (eps * T.exp(sigma))  # calculate latent variables
        return z

    def get_output_shape_for(self, input_shapes):
        if input_shapes[0] != input_shapes[1]:
            raise ValueError('Both input layers should have the same shape.')
        return input_shapes[0]


if __name__ == "__main__":
    # model parameters
    minibatch_size = 100
    input_dim = 784
    # number of hidden units in encoder (x -> z) network
    encoder_hidden_dim = 500
    # number of hidden units in decoder (z -> x) network
    decoder_hidden_dim = 500
    # number of latent variables
    latent_dim = 2  # pairs of mu and sigma

    # l2 regularization weight
    lamda = 0.001
    # learning rate
    learning_rate = 0.02

    # random number generator used for sampling latent variables
    srng = RandomStreams(seed=123)

    # input to the network
    x = T.fmatrix(name='x')

    # build the model
    l_input = lasagne.layers.InputLayer(shape=(None, input_dim), input_var=x)

    l_encoder_hidden = lasagne.layers.DenseLayer(l_input, num_units=encoder_hidden_dim,
                                                 W=lasagne.init.Normal(0.01), b=lasagne.init.Normal(0.01),
                                                 nonlinearity=lasagne.nonlinearities.tanh)

    l_encoder_mu = lasagne.layers.DenseLayer(l_encoder_hidden, num_units=latent_dim,
                                             W=lasagne.init.Normal(0.01), b=lasagne.init.Normal(0.01),
                                             nonlinearity=lasagne.nonlinearities.identity)

    l_encoder_sigma = lasagne.layers.DenseLayer(l_encoder_hidden, num_units=latent_dim,
                                                W=lasagne.init.Normal(0.01), b=lasagne.init.Normal(0.01),
                                                nonlinearity=lasagne.nonlinearities.identity)

    l_latent = NormalSamplingLayer(incomings=[l_encoder_mu, l_encoder_sigma], rng=srng)

    l_decoder_hidden = lasagne.layers.DenseLayer(l_latent, num_units=decoder_hidden_dim,
                                                 W=lasagne.init.Normal(0.01), b=lasagne.init.Normal(0.01),
                                                 nonlinearity=lasagne.nonlinearities.tanh)

    l_decoder_output = lasagne.layers.DenseLayer(l_decoder_hidden, num_units=input_dim,
                                                 W=lasagne.init.Normal(0.01), b=lasagne.init.Normal(0.01),
                                                 nonlinearity=lasagne.nonlinearities.sigmoid)

    mu = lasagne.layers.get_output(l_encoder_mu)
    sigma = lasagne.layers.get_output(l_encoder_sigma)

    y = lasagne.layers.get_output(l_decoder_output)

    # LOSS FUNCTION
    # first KL term in Eqn. 10 in the paper. Acts as a regularization on latent variables.
    ll_bound_kl_term = 0.5 * (mu.shape[0]*mu.shape[1] + T.sum(2. * sigma) -
                              T.sum(T.square(mu)) -
                              T.sum(T.exp(2. * sigma))) / minibatch_size

    # second term in Eqn. 10, measures data fit. Note because the outputs are 0-1, we use binary crossentropy.
    ll_bound_fit_term = T.sum(-1. * lasagne.objectives.binary_crossentropy(y, x)) / minibatch_size

    # likelihood bound
    ll_bound = ll_bound_kl_term + ll_bound_fit_term

    # L2 regularization
    l2_cost = lasagne.regularization.regularize_network_params(l_decoder_output, lasagne.regularization.l2)

    # Loss
    loss = -(ll_bound_kl_term + ll_bound_fit_term) + (lamda * l2_cost)

    model_params = lasagne.layers.get_all_params(l_decoder_output)

    gd_updates = lasagne.updates.adagrad(loss, model_params, learning_rate=learning_rate)

    # TRAINING

    # number of training epochs, i.e., passes over training set.
    # there are 50.000 samples in training set. 500 epochs means training over 25.000.000 samples. In the paper, they
    # keep training for much longer (around 100.000.000)
    epoch_count = 500

    # report log_likelihood bound on training after this many training samples
    report_interval = 25000

    # load the training data
    (tx, ty), (vx, vy), (_, _) = load_mnist(path='../datasets')
    train_n = tx.shape[0]
    val_n = vx.shape[0]
    # we load the data into shared variables, this is recommended if you do training on GPU.
    train_x = theano.shared(np.asarray(tx, dtype=np.float32))
    val_x = theano.shared(np.asarray(vx, dtype=np.float32))

    # index of the first sample in batch
    batch_start = T.lscalar()

    # training and validation functions
    train_model = theano.function([batch_start], ll_bound, updates=gd_updates,
                                  givens={x: train_x[batch_start:(batch_start+minibatch_size)]})

    validate_model = theano.function([batch_start], ll_bound,
                                     givens={x: val_x[batch_start:(batch_start+minibatch_size)]})

    # arrays for keeping track of log ll bounds on training and validation sets
    train_ll = []
    val_ll = []

    # TRAINING LOOP
    for e in range(epoch_count):
        ll = 0.0
        for i in range(0, train_n, minibatch_size):
            cll = train_model(i)  # one step of gradient descent
            ll += cll
            # report log ll bound
            if (i + minibatch_size) % report_interval == 0:
                ll = (ll * minibatch_size) / report_interval
                print "|Epoch {0:d}|\tTrain ll: {1:f}".format(e, ll)
                train_ll.append(ll)
                ll = 0.0

        # epoch over, shuffle training data
        train_x = theano.shared(np.asarray(tx[np.random.permutation(train_n)], dtype=np.float32))

        # calculate log ll on validation and report it
        ll = 0.0
        for i in range(0, val_n, minibatch_size):
            cll = validate_model(i)
            ll += cll

        ll = (ll * minibatch_size) / val_n
        print "|Epoch {0:d}|\tVal ll: {1:f}".format(e, ll)
        val_ll.append(ll)

    print("Training over.")

    if latent_dim == 2:
        print("Generating latent space figure")
        img = np.zeros((28*20, 28*20))

        # function for calculating x given z
        z = T.fvector()
        generate_x = lasagne.layers.get_output(l_decoder_output, {l_latent: z})
        generate_fn = theano.function([z], generate_x)

        # calculate min and max latent variable values over validation set
        latent = lasagne.layers.get_output(l_encoder_mu)
        calc_latent = theano.function([x], latent)
        val_z = calc_latent(vx)
        minz, maxz = np.min(val_z), np.max(val_z)

        zvals = np.linspace(minz, maxz, 20)
        for i, z1 in enumerate(zvals):
            for j, z2 in enumerate(zvals):
                xp = generate_fn(np.asarray([z1, z2], dtype=np.float32))
                img[(i*28):((i+1)*28), (j*28):((j+1)*28)] = xp.reshape((28, 28))

        """
        import matplotlib.pyplot as plt
        plt.style.use('classic')
        plt.imshow(img, cmap='gray')
        """

        from scipy import misc
        misc.imsave('Fig4b.png', img)


