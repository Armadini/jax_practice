import tensorflow as tf

from typing import Tuple, Iterable
import jax.numpy as jnp
from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np
from jax import grad, value_and_grad
from jax import numpy as jnp
import jax
import cv2
import numpy as np

keys = jax.random.split(jax.random.PRNGKey(seed=42), 10)

# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
# print(x_train.shape)

# img_fin = cv2.resize(x_train, (227, 227))

dataset = jax.random.normal(
            keys[0], (10, 11, 11, 3))

conv1_filters = jax.random.normal(
            keys[1], (96, 11, 11, 3))

print(dataset.shape)
print(conv1_filters.shape)

prod = np.tensordot(dataset, conv1_filters, axes=([1,2,3], [1,2,3]))
print(prod.shape)
res = np.sum(prod, axis=1)
print(res.shape)