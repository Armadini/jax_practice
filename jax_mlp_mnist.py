from typing import Tuple, Iterable
import jax.numpy as jnp
from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np
from jax import grad, value_and_grad
from jax import numpy as jnp
import jax
import tensorflow as tf
from alexnet_params import AlexNet_params, flatten_AlexNet_params, unflatten_AlexNet_params

print("JAX Version : {}".format(jax.__version__))


# Define API

def Relu(x):
    return jnp.maximum(x, jnp.zeros_like(x)) # max(0,x)

def linear_layer(weights, input_data, activation=lambda x: x):
    w, b = weights
    out = jnp.dot(input_data, w.T) + b
    return activation(out)


def forward_pass(params, x, y):
    layer_output = x

    for p in params[:-1]:
        layer_output = linear_layer(p, layer_output, Relu)

    return linear_layer(params[-1], layer_output, jax.nn.softmax)


def mse(params, x, y):
    preds = forward_pass(params, x, y)
    # print(f"Params from MSE {params}")
    return jnp.square(preds - y).mean()

# Jax callable function object.
# This will be important. The constructor will have to create our entire model (functions/operations)
# Will probably also move the params constructor in here...
def model(params, x, y):
    lr = .005
    # Forward pass + Calculate loss
    # loss = mse(params, x, y, rand_key)
    # Calculate Gradients
    loss, grads = value_and_grad(mse, 0)(params, x, y)
    # Update params + return them
    update_fun = lambda p, g: p - (g*.001)
    params = jax.tree_multimap(update_fun, params, grads)
    return params, loss


num_epochs = 100
rand_key = jax.random.PRNGKey(42)
keys = jax.random.split(rand_key, 10)

# Dataset in appropriate shape
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

# print("X:", x)
# print("Y:", y)

# Some type of pytree that represents our models shape/structure
params = [
    [jax.random.normal(keys[1], (1400,784)),
        jax.random.normal(keys[2], (1400,))],
    [jax.random.normal(keys[3], (1000,1400)),
        jax.random.normal(keys[4], (1000,))],
    [jax.random.normal(keys[5], (500,1000)),
        jax.random.normal(keys[6], (500,))],
    [jax.random.normal(keys[7], (100,500)),
        jax.random.normal(keys[8], (100,))],
    [jax.random.normal(keys[7], (50,100)),
        jax.random.normal(keys[8], (50,))],
    [jax.random.normal(keys[7], (10,50)),
        jax.random.normal(keys[8], (10,))],
]

for epoch in range(num_epochs):
    loss = 1
    for x_batch, y_batch in zip(x_test, y_test):
        # This line should do a forward pass, calculate gradients, update weights -> returns new params
        params, loss = model(params, x_batch, y_batch)
    print(f"Loss at epoch {epoch}: {loss}")
        # print(f"Params {params}")
