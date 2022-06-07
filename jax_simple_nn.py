import jax

print("JAX Version : {}".format(jax.__version__))

from jax import numpy as jnp
from jax import grad, value_and_grad

import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split
from jax import numpy as jnp

X, Y = datasets.load_boston(return_X_y=True)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, random_state=123)

X_train, X_test, Y_train, Y_test = jnp.array(X_train, dtype=jnp.float32),\
                                   jnp.array(X_test, dtype=jnp.float32),\
                                   jnp.array(Y_train, dtype=jnp.float32),\
                                   jnp.array(Y_test, dtype=jnp.float32)

samples, features = X_train.shape

X_train.shape, X_test.shape, Y_train.shape, Y_test.shape

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std


def InitializeWeights(layer_sizes, seed):
    weights = []

    for i, units in enumerate(layer_sizes):
        if i==0:
            w = jax.random.uniform(key=seed, shape=(units, features), minval=-1.0, maxval=1.0, dtype=jnp.float32)
        else:
            w = jax.random.uniform(key=seed, shape=(units, layer_sizes[i-1]), minval=-1.0, maxval=1.0,
                                   dtype=jnp.float32)

        b = jax.random.uniform(key=seed, minval=-1.0, maxval=1.0, shape=(units,), dtype=jnp.float32)

        weights.append([w,b])

    return weights



seed = jax.random.PRNGKey(123)
weights = InitializeWeights([5,10,1], seed)

for w in weights:
    print(w[0].shape, w[1].shape)

def Relu(x):
    return jnp.maximum(x, jnp.zeros_like(x)) # max(0,x)

x = jnp.array([-1,0,1,-2,4,-6,5])

Relu(x)


def LinearLayer(weights, input_data, activation=lambda x: x):
    w, b = weights
    out = jnp.dot(input_data, w.T) + b
    return activation(out)


rand_data = jax.random.uniform(key=seed, shape=(5, X_train.shape[1]))

out = LinearLayer(weights[0], rand_data)

print("Data Shape : {}".format(rand_data.shape))
print("Output Shape : {}".format(out.shape))


def ForwardPass(weights, input_data):
    layer_out = input_data

    for i in range(len(weights[:-1])):
        layer_out = LinearLayer(weights[i], layer_out, Relu)

    preds = LinearLayer(weights[-1], layer_out)

    return preds.squeeze()


preds = ForwardPass(weights, X_train)

preds.shape


def MeanSquaredErrorLoss(weights, input_data, actual):
    preds = ForwardPass(weights, input_data)
    return jnp.power(actual - preds, 2).mean()


from jax import grad, value_and_grad

def CalculateGradients(weights, input_data, actual):
    Grad_MSELoss = grad(MeanSquaredErrorLoss)
    gradients = Grad_MSELoss(weights, input_data, actual)
    return gradients


def TrainModel(weights, X, Y, learning_rate, epochs):
    for i in range(epochs):
        loss = MeanSquaredErrorLoss(weights, X, Y)
        gradients = CalculateGradients(weights, X, Y)

        ## Update Weights
        for j in range(len(weights)):
            weights[j][0] -= learning_rate * gradients[j][0] ## Update Weights
            weights[j][1] -= learning_rate * gradients[j][1] ## Update Biases

        if i%100 ==0: ## Print MSE every 100 epochs
            print("MSE : {:.2f}".format(loss))


seed = jax.random.PRNGKey(42)
learning_rate = jnp.array(1/1e3)
epochs = 1500
layer_sizes = [5,10,15,1]

weights = InitializeWeights(layer_sizes, seed)

TrainModel(weights, X_train, Y_train, learning_rate, epochs)

test_preds = ForwardPass(weights, X_test)

test_preds[:5], Y_test[:5]

train_preds = ForwardPass(weights, X_train)

train_preds[:5], Y_train[:5]


print("Test  MSE Score : {:.2f}".format(MeanSquaredErrorLoss(weights, X_test, Y_test)))
print("Train MSE Score : {:.2f}".format(MeanSquaredErrorLoss(weights, X_train, Y_train)))

from sklearn.metrics import r2_score

print("Test  R^2 Score : {:.2f}".format(r2_score(test_preds.squeeze(), Y_test)))
print("Train R^2 Score : {:.2f}".format(r2_score(train_preds.squeeze(), Y_train)))

