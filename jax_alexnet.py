from typing import Tuple, Iterable
import jax.numpy as jnp
from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np
from jax import grad, value_and_grad
from jax import numpy as jnp
import jax
from alexnet_params import AlexNet_params, flatten_AlexNet_params, unflatten_AlexNet_params

print("JAX Version : {}".format(jax.__version__))


# Define API

jax.tree_util.register_pytree_node(
    AlexNet_params, flatten_func=flatten_AlexNet_params, unflatten_func=unflatten_AlexNet_params
)

# Dataset in appropriate shape
dataset = None
# Inputs with shape (n,227,227,3) but for now (n, 64)
x = []
# Outputs with shape (n, 10)
y = []


def linear_layer(weights, input_data, activation=lambda x: x):
    w, b = weights
    out = jnp.dot(input_data, w.T) + b
    return activation(out)

def conv_layer(filter, input_data, , activation=lambda x: x):

def forward_pass(params, x, y, rand_key):
    return None

def mse(params, x, y, rand_kek):
    preds = forward_pass(params, x, y, rand_key)
    return jnp.square(preds - y).mean()

# Jax callable function object.
# This will be important. The constructor will have to create our entire model (functions/operations)
# Will probably also move the params constructor in here...


def model(params, x, y, rand_key):
    
    # Forward pass + Calculate loss

    # Calculate Gradients

    # Update params + return them
    return None


num_epochs = 10
rand_key = jax.random.PRNGKey(42)

# Some type of pytree that represents our models shape/structure
params = AlexNet_params(rand_key)

for epoch in num_epochs:
    print("information...")
    for x_batch, y_batch in zip(x, y):
        # This line should do a forward pass, calculate gradients, update weights -> returns new params
        params = model(params, x_batch, y_batch, rand_key)
