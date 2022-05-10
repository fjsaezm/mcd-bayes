# Implements auto-encoding variational Bayes.

import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.norm as norm

from autograd import grad
from data import load_mnist
from data import save_images as s_images
from autograd.misc import (
    flatten,
)  # This is used to flatten the params (transforms a list into a numpy array)

# images is an array with one row per image, file_name is the png file on which to save the images


def save_images(images, file_name):
    return s_images(images, file_name, vmin=0.0, vmax=1.0)


# Sigmoid activiation function to estimate probabilities


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# Relu activation function for non-linearity


def relu(x):
    return np.maximum(0, x)


# This function intializes the parameters of a deep neural network


def init_net_params(layer_sizes, scale=1e-2):

    """Build a (weights, biases) tuples for all layers."""
    # layer_sizes is a list with the sizes of each layer

    return [
        (scale * npr.randn(m, n), scale * npr.randn(n))  # weight matrix  # bias vector
        for m, n in zip(layer_sizes[:-1], layer_sizes[1:])
    ]


# This will be used to normalize the activations of the NN

# This computes the output of a deep neuralnetwork with params a list with pairs of weights and biases


def neural_net_predict(params, inputs):

    """Params is a list of (weights, bias) tuples.
    inputs is an (N x D) matrix.
    Applies relu to every layer but the last."""

    for W, b in params[:-1]:
        outputs = np.dot(inputs, W) + b  # linear transformation
        inputs = relu(outputs)  # nonlinear transformation

    # Last layer is linear

    outW, outb = params[-1]
    outputs = np.dot(inputs, outW) + outb

    return outputs


# This implements the reparametrization trick


def sample_latent_variables_from_posterior(encoder_output):
    # TODO use the reparametrization trick to generate one sample from q(z|x) per each batch datapoint
    # use npr.randn for that.
    # The output of this function is a matrix of size the batch x the number of latent dimensions

    # Params of a diagonal Gaussian.
    D = np.shape(encoder_output)[-1] // 2
    mean, log_std = encoder_output[:, :D], encoder_output[:, D:]

    # This is equation 15 form the given assignment
    sampled = mean + np.multiply(np.exp(log_std), np.random.randn(mean.shape[0], mean.shape[1]))

    return sampled


# This evlauates the log of the term that depends on the data


def bernoulli_log_prob(targets, logits):
    # TODO compute the log probability of the targets given the
    # generator output specified in logits
    # sum the probabilities across the dimensions of each image in the batch.
    # The output of this function should be a vector of size the batch size

    # logits are in R, this is the output of the neuroal network, f(z).
    # Targets must be between 0 and 1, these are the observations, x.

    # Pre-evaluate the sigmoid
    eval_sigmoid = sigmoid(logits)

    # This is equation 3 form the given assignment
    bernuilli_log_probs = [
        np.prod(
            np.multiply(targets_row, eval_sigmoid_row)
            + np.multiply(1 - targets_row, 1 - eval_sigmoid_row)
        )
        for targets_row, eval_sigmoid_row in zip(targets, eval_sigmoid)
    ]

    return bernuilli_log_probs


# This evaluates the KL between q and the prior


def compute_KL(q_means_and_log_stds):
    # TODO compute the KL divergence between q(z|x)
    # and the prior (use a standard Gaussian for the prior)
    # Use the fact that the KL divervence is the sum of KL divergence
    # of the marginals if q and p factorize
    # The output of this function should be a vector of size the batch size

    D = np.shape(q_means_and_log_stds)[-1] // 2
    mean, log_std = q_means_and_log_stds[:, :D], q_means_and_log_stds[:, D:]

    # This is equation 12 form the given assignment
    kl_divergence = np.array([
        0.5 * np.sum(np.exp(log_std_row) + np.square(mean_row)**2 - 1 - log_std_row)
        for mean_row, log_std_row in zip(mean, log_std)
    ])

    return kl_divergence


# This evaluates the lower bound


def vae_lower_bound(gen_params, rec_params, data, N):
    # TODO compute a noisy estimate of the lower bound by using a single Monte Carlo sample:

    # 1 - compute the encoder output using neural_net_predict given the data and rec_params
    encoder_output = neural_net_predict(params=rec_params, inputs=data)

    # 2 - sample the latent variables associated to the batch in data
    #     (use sample_latent_variables_from_posterior and the encoder output)
    sampled_latent_variable = sample_latent_variables_from_posterior(encoder_output)

    # 3 - use the sampled latent variables to reconstruct the image and to compute the log_prob of the actual data
    #     (use neural_net_predict for that)
    decoder_output = neural_net_predict(
        params=gen_params, inputs=sampled_latent_variable
    )

    # 4 - compute the KL divergence between q(z|x) and the prior (use compute_KL for that)
    kl_divergence = compute_KL(encoder_output)

    # 5 - return an average estimate (per batch point) of the lower bound
    # by substracting the KL to the data dependent term
    # This is equation 12 form the given assignment
    lower_bound_estimate = N * np.mean(np.log(decoder_output) - kl_divergence.reshape(-1,1))

    return lower_bound_estimate


if __name__ == "__main__":

    # Model hyper-parameters
    npr.seed(2**32 - 1)  # We fix the random seed for reproducibility

    latent_dim = 50
    data_dim = 784  # How many pixels in each image (28x28).
    n_units = 200
    n_layers = 2

    gen_layer_sizes = [latent_dim] + [n_units for i in range(n_layers)] + [data_dim]
    rec_layer_sizes = [data_dim] + [n_units for i in range(n_layers)] + [latent_dim * 2]

    # Training parameters
    batch_size = 200
    num_epochs = 30
    learning_rate = 0.001

    # ADAM parameters
    alpha = 1e-3
    beta_1 = 0.9
    beta_2 = 0.999
    epsilon = 1e-8

    print("Loading training data...")
    N, train_images, _, test_images, _ = load_mnist()

    # Parameters for the generator network p(x|z)
    init_gen_params = init_net_params(gen_layer_sizes)

    # Parameters for the recognition network p(z|x)
    init_rec_params = init_net_params(rec_layer_sizes)
    combined_params_init = (init_gen_params, init_rec_params)
    num_batches = int(np.ceil(len(train_images) / batch_size))

    # We flatten the parameters (transform the lists or tupples into numpy arrays)
    flattened_combined_params_init, unflat_params = flatten(combined_params_init)

    # Actual objective to optimize that receives flattened params
    def objective(flattened_combined_params):
        combined_params = unflat_params(flattened_combined_params)
        data_idx = batch
        gen_params, rec_params = combined_params

        # We binarize the data
        on = train_images[data_idx, :] > npr.uniform(
            size=train_images[data_idx, :].shape
        )
        images = train_images[data_idx, :] * 0.0
        images[on] = 1.0

        return vae_lower_bound(gen_params, rec_params, images, N)

    # Get gradients of objective using autograd.
    objective_grad = grad(objective)
    flattened_current_params = flattened_combined_params_init

    # ADAM parameters
    t = 1

    # TODO write here the initial values for the ADAM parameters (including the m and v vectors)
    # you can use np.zeros_like(flattened_current_params) to initialize m and v
    m = np.zeros_like(flattened_current_params)
    v = np.zeros_like(flattened_current_params)

    # We do the actual training - ADAM optimization
    for epoch in range(num_epochs):
        elbo_est = 0.0

        #print('before:', flattened_current_params[:3])
        for n_batch in range(int(np.ceil(N / batch_size))):
            batch = np.arange(
                batch_size * n_batch, np.minimum(N, batch_size * (n_batch + 1))
            )

            # Compute the noisy gradients
            grad = objective_grad(flattened_current_params)
            #print('grad:', grad[:3])
            #print('n_batch - {}:'.format(n_batch), flattened_current_params[:3])

            # TODO Use the estimated noisy gradient in grad to update the paramters using the ADAM updates
            elbo_est += objective(flattened_current_params)

            # ADAM step
            m = beta_1 * m + (1 - beta_1) * grad
            v = beta_2 * v + (1 - beta_2) * np.square(grad)**2
            hat_m = m / (1 - beta_1**t)
            hat_v = v / (1 - beta_2**t)
            flattened_current_params -= np.divide(
                alpha * hat_m, np.sqrt(hat_v) + epsilon
            )

            t += 1

        print("Epoch: %d ELBO: %e" % (epoch, elbo_est / np.ceil(N / batch_size)))

    # We obtain the final trained parameters
    gen_params, rec_params = unflat_params(flattened_current_params)

    # ----- TASK 3.1 ----
    # TODO Generate 25 images from prior (use neural_net_predict) and save them using save_images
    path_created_images = "data/3_1.png"
    n_images = 25

    # Sample z form prior, N(0,1)
    noise_images = [npr.randn(28, 28) for _ in range(n_images)]

    # Apply autoencoder with noise iamges
    encoder_output = neural_net_predict(gen_params, noise_images)
    sampled_latent_variable = sample_latent_variables_from_posterior(encoder_output)
    created_images = neural_net_predict(rec_params, sampled_latent_variable)

    # Concatenate in a single image
    big_img = np.array([])
    for i in range(0, 21, 5):
        row = np.concatenate(created_images[i:i+5], axis=0)
        big_img = np.append(big_img, row.reshape((1,-1)))
    big_img = big_img.reshape((28*5, 28*5))

    # Save image
    save_images(big_img, path_created_images)

    print('3.1 Done!')

    # TODO Generate image reconstructions for the first 10 test images (use neural_net_predict for each model)
    # and save them alongside with the original image using save_images

    num_interpolations = 5
    for i in range(5):

        # TODO Generate 5 interpolations from the first test image to the second test image,
        # for the third to the fourth and so on until 5 interpolations
        # are computed in latent space and save them using save images.
        # Use a different file name to store the images of each iterpolation.
        # To interpolate from  image I to image G use a convex conbination. Namely,
        # I * s + (1-s) * G where s is a sequence of numbers from 0 to 1 obtained by numpy.linspace
        # Use mean of the recognition model as the latent representation.

        pass
