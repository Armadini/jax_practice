from typing import Tuple, Iterable
import jax.numpy as jnp
from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np
from jax import grad, value_and_grad
from jax import numpy as jnp
import jax

print("JAX Version : {}".format(jax.__version__))


# Define API


class AlexNet_params:

    def __init__(self, rand_key, conv1_filters=None, conv2_filters=None, conv3_filters=None, conv4_filters=None, conv5_filters=None, fc1=None, fc2=None, fc3=None):
        self.keys = jax.random.split(rand_key, 10)
        self.conv1_filters = jax.random.normal(
            self.keys[1], (96, 11, 11, 3)) if conv1_filters == None else conv1_filters
        self.conv2_filters = jax.random.normal(
            self.keys[2], (256, 5, 5, 96)) if conv2_filters == None else conv2_filters
        self.conv3_filters = jax.random.normal(
            self.keys[3], (384, 3, 3, 256)) if conv3_filters == None else conv3_filters
        self.conv4_filters = jax.random.normal(
            self.keys[4], (384, 3, 3, 384)) if conv4_filters == None else conv4_filters
        self.conv5_filters = jax.random.normal(
            self.keys[5], (256, 3, 3, 384)) if conv5_filters == None else conv5_filters

        self.fc1 = jax.random.normal(
            self.keys[6], (4096,)) if fc1 == None else fc1
        self.fc2 = jax.random.normal(
            self.keys[7], (4096,)) if fc2 == None else fc2
        self.fc3 = jax.random.normal(
            self.keys[8], (1000,)) if fc3 == None else fc3



def flatten_AlexNet_params(alexNet_params) -> Tuple[Iterable[int], str]:
    flat_contents = [
        alexNet_params.conv1_filters,
        alexNet_params.conv2_filters,
        alexNet_params.conv3_filters,
        alexNet_params.conv4_filters,
        alexNet_params.conv5_filters,
        alexNet_params.fc1,
        alexNet_params.fc2,
        alexNet_params.fc3
    ]

    return flat_contents, alexNet_params.keys


def unflatten_AlexNet_params(aux_data, flat_contents):
    return AlexNet_params(aux_data, *flat_contents)


if __name__ == "__main__":
    jax.tree_util.register_pytree_node(
        AlexNet_params, flatten_func=flatten_AlexNet_params, unflatten_func=unflatten_AlexNet_params
    )

    rand_key = jax.random.PRNGKey(42)
    leaves = jax.tree_leaves(AlexNet_params(rand_key=rand_key))
    for layer in leaves:
        np_arr = jnp.array(layer)
        print(np_arr.shape)
