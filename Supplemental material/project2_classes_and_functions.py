import numpy as np

# Functions:

def Design(x, p):
    """Create a design matrix for polynomial regression."""
    X = np.ones((x.shape[0], p + 1))
    for i in range(1, p + 1):
        X[:, i] = x[:, 0] ** i
    return X

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

# Defining some activation functions
def ReLU(z):
    return np.where(z > 0, z, 0)

# Derivative of the ReLU function
def ReLU_der(z):
    return np.where(z > 0, 1, 0)

# Leaky ReLU and its derivative
def leaky_ReLU(x, alpha=0.01):

    return np.where(x > 0, x, alpha * x)

def der_leaky_ReLU(x, alpha=0.01):

    return np.where(x > 0, 1, alpha)

def mean_squared_error(predictions, targets):
    return np.mean((predictions - targets) ** 2)

def mean_squared_error_derivative(predictions, targets):
    return 2 * (predictions - targets) / targets.size

def linear(x):
    return x  # Linear activation

def linear_derivative(x):
    return np.ones_like(x)  # Derivative of linear is 1

def R2(z,zpred):
    return 1 - np.sum((z - zpred)**2) / np.sum((z - np.mean(z)) ** 2)

def CostLogReg(target):

    def func(X):

        return -(1.0 / target.shape[0]) * np.sum(
            (target * np.log(X + 10e-10)) + ((1 - target) * np.log(1 - X + 10e-10))
        )

    return func

# Log loss function:

def CostCrossEntropy(predictions, targets):
    return -(1.0 / targets.size) * np.sum(targets * np.log(predictions + 1e-10))

def CostCrossEntropyDer(predictions, targets):
    predictions = np.clip(predictions, 1e-10, 1 - 1e-10)
    return -(targets / predictions) + (1 - targets) / (1 - predictions)

class NetworkClass:
    def __init__(self, cost_fun, cost_der, network_input_size, layer_output_sizes, activation_funcs, activation_ders):
        self.cost_fun = cost_fun
        self.cost_der = cost_der
        self.layers = []
        self.activation_funcs = activation_funcs
        self.activation_ders = activation_ders

        # Architecture check
        # print("Architecture configuration:")
        # print("--------------------------")
        # print("Network input size:", network_input_size)
        # print("Layer output sizes:", layer_output_sizes)
        # print("Activation functions:", activation_funcs)
        # print("Activation derivatives:", activation_ders)

        input_size = network_input_size
        for i, output_size in enumerate(layer_output_sizes):
            self.layers.append({
                'weights': np.random.randn(input_size, output_size) * np.sqrt(2. / input_size),
                'biases': np.zeros((1, output_size))
            })
            input_size = output_size

    def predict(self, inputs):
        activations, _, _ = self._feed_forward_saver(inputs)
        return activations[-1]  # Return the final activation

    def cost(self, inputs, targets):
        predictions = self.predict(inputs)
        return self.cost_fun(predictions, targets)

    def _feed_forward_saver(self, inputs):
        activations = [inputs]
        zs = []
        layer_inputs = []

        for layer in self.layers:
            layer_inputs.append(activations[-1])
            z = np.dot(activations[-1], layer['weights']) + layer['biases']
            a = self.activation_funcs[len(activations) - 1](z)
            zs.append(z)
            activations.append(a)

        return activations, zs, layer_inputs

    def compute_gradient(self, inputs, targets):
        activations, zs, layer_inputs = self._feed_forward_saver(inputs)
        layer_grads = []
        output = activations[-1]

        delta = self.cost_der(output, targets) * self.activation_ders[len(self.layers) - 1](zs[-1])

        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            prev_activation = layer_inputs[i]
            layer_grads.append({
                'weights': np.dot(prev_activation.T, delta),
                'biases': np.sum(delta, axis=0, keepdims=True)
            })
            if i > 0:
                delta = np.dot(delta, layer['weights'].T) * self.activation_ders[i - 1](zs[i - 1])

        layer_grads.reverse()
        return layer_grads

    def update_weights(self, layer_grads, learning_rate, lmbd):
        for i, layer in enumerate(self.layers):
            layer['weights'] -= learning_rate * (layer_grads[i]['weights'] + lmbd * layer['weights'])
            layer['biases'] -= learning_rate * layer_grads[i]['biases']

    def train(self, X, y, epochs, batch_size, learning_rate, lmbd):
        for epoch in range(epochs):
            # Shuffle data
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            # Mini-batch training
            for start in range(0, X.shape[0], batch_size):
                end = min(start + batch_size, X.shape[0])
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                layer_grads = self.compute_gradient(X_batch, y_batch)
                self.update_weights(layer_grads, learning_rate, lmbd)

            # Optional: Print cost at the end of each epoch
            epoch_cost = self.cost(X, y)
            #print(f"Epoch {epoch + 1}/{epochs}, Cost: {epoch_cost}")




