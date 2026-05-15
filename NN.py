import numpy as np

np.random.seed(0)

# ---------------------
# Load Data
# ---------------------
data = np.load("mnist.npz")

images = data['images'].reshape(data['images'].shape[0], -1).astype(np.float32)
X = (images - 0.5) * 2   # normalize to [-1, 1]
y = data['labels']

X_train, y_train = X[:8000], y[:8000]
X_test, y_test = X[8000:], y[8000:]

# ---------------------
# Dense Layer
# ---------------------
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2.0 / n_inputs)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        samples = len(self.inputs)
        self.dweights = np.dot(self.inputs.T, dvalues) / samples
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True) / samples
        self.dinputs = np.dot(dvalues, self.weights.T)

# ---------------------
# ReLU
# ---------------------
class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

# ---------------------
# Softmax + Loss
# ---------------------
class Activation_Softmax_Loss_CategoricalCrossentropy:

    def forward(self, inputs, y_true):
        self.y_true = y_true
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        samples = len(inputs)
        correct_confidences = self.output[range(samples), y_true]
        loss = -np.log(correct_confidences + 1e-7)
        return np.mean(loss)

    def backward(self):
        samples = len(self.output)

        self.dinputs = self.output.copy()
        self.dinputs[range(samples), self.y_true] -= 1
        self.dinputs = self.dinputs / samples

# ---------------------
# Adam Optimizer
# ---------------------
class Optimizer_Adam:
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7):
        self.lr = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.iterations = 0

    def update_params(self, layer):
        if not hasattr(layer, 'm_w'):
            layer.m_w = np.zeros_like(layer.weights)
            layer.v_w = np.zeros_like(layer.weights)
            layer.m_b = np.zeros_like(layer.biases)
            layer.v_b = np.zeros_like(layer.biases)

        self.iterations += 1

        # Momentum
        layer.m_w = self.beta_1 * layer.m_w + (1 - self.beta_1) * layer.dweights
        layer.m_b = self.beta_1 * layer.m_b + (1 - self.beta_1) * layer.dbiases

        # RMSProp
        layer.v_w = self.beta_2 * layer.v_w + (1 - self.beta_2) * (layer.dweights ** 2)
        layer.v_b = self.beta_2 * layer.v_b + (1 - self.beta_2) * (layer.dbiases ** 2)

        # Bias correction
        m_w_corr = layer.m_w / (1 - self.beta_1 ** self.iterations)
        m_b_corr = layer.m_b / (1 - self.beta_1 ** self.iterations)

        v_w_corr = layer.v_w / (1 - self.beta_2 ** self.iterations)
        v_b_corr = layer.v_b / (1 - self.beta_2 ** self.iterations)

        # Update
        layer.weights -= self.lr * m_w_corr / (np.sqrt(v_w_corr) + self.epsilon)
        layer.biases  -= self.lr * m_b_corr / (np.sqrt(v_b_corr) + self.epsilon)

# ---------------------
# Create Network
# ---------------------
layer1 = Layer_Dense(784, 256)
activation1 = Activation_ReLU()

layer2 = Layer_Dense(256, 128)
activation2 = Activation_ReLU()

layer3 = Layer_Dense(128, 10)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

optimizer = Optimizer_Adam(learning_rate=0.001)

# ---------------------
# Training
# ---------------------
batch_size = 64

for epoch in range(51):

    # shuffle (without overwriting original)
    indices = np.arange(len(X_train))
    np.random.shuffle(indices)
    X_shuffled = X_train[indices]
    y_shuffled = y_train[indices]

    epoch_loss = 0
    steps = 0

    for i in range(0, len(X_train), batch_size):
        X_batch = X_shuffled[i:i+batch_size]
        y_batch = y_shuffled[i:i+batch_size]

        # forward
        layer1.forward(X_batch)
        activation1.forward(layer1.outputs)

        layer2.forward(activation1.outputs)
        activation2.forward(layer2.outputs)

        layer3.forward(activation2.outputs)
        loss = loss_activation.forward(layer3.outputs, y_batch)

        epoch_loss += loss
        steps += 1

        # backward
        loss_activation.backward()
        layer3.backward(loss_activation.dinputs)

        activation2.backward(layer3.dinputs)
        layer2.backward(activation2.dinputs)

        activation1.backward(layer2.dinputs)
        layer1.backward(activation1.dinputs)

        # update
        optimizer.update_params(layer1)
        optimizer.update_params(layer2)
        optimizer.update_params(layer3)

    epoch_loss /= steps

    # -------- evaluation (TRAIN) --------
    layer1.forward(X_train)
    activation1.forward(layer1.outputs)

    layer2.forward(activation1.outputs)
    activation2.forward(layer2.outputs)

    layer3.forward(activation2.outputs)

    exp_values = np.exp(layer3.outputs - np.max(layer3.outputs, axis=1, keepdims=True))
    probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

    predictions = np.argmax(probabilities, axis=1)
    accuracy = np.mean(predictions == y_train)

    print(f"epoch {epoch} | loss: {epoch_loss:.3f} | acc: {accuracy:.3f}")

# ---------------------
# Test accuracy
# ---------------------
layer1.forward(X_test)
activation1.forward(layer1.outputs)

layer2.forward(activation1.outputs)
activation2.forward(layer2.outputs)

layer3.forward(activation2.outputs)

exp_values = np.exp(layer3.outputs - np.max(layer3.outputs, axis=1, keepdims=True))
probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

predictions = np.argmax(probabilities, axis=1)
accuracy = np.mean(predictions == y_test)

print(f"Test accuracy: {accuracy:.3f}")

## final test
def forprop(canvas_flat):
    # Convert to NumPy array if it’s a list
    x = np.array(canvas_flat, dtype=np.float32)

    # Normalize like training
    x = (x - 0.5) * 2

    # Ensure shape is (1, 784)
    x = x.reshape(1, -1)

    # Forward pass
    layer1.forward(x)
    activation1.forward(layer1.outputs)

    layer2.forward(activation1.outputs)
    activation2.forward(layer2.outputs)

    layer3.forward(activation2.outputs)

    # Softmax
    exp_values = np.exp(layer3.outputs - np.max(layer3.outputs, axis=1, keepdims=True))
    probs = exp_values / np.sum(exp_values, axis=1, keepdims=True)

    print("probs:", probs)  # optional debug

    # Return single predicted digit
    return int(np.argmax(probs))

import json

while True:
    f = input("data: ")
    print("##################################################################################")
    data = json.loads(f)  # converts string → list
    print("----------------------------------------------------------------------------------")
    print("Your Number:",forprop(data))