from Neuron import Neuron
import numpy as np


class Layer:
    def __init__(self, input_size, neuron_count, use_bias=True):
        self.neurons = [Neuron(input_size, use_bias) for _ in range(neuron_count)]
        self.input_size = input_size
        self.neuron_count = neuron_count
        self.inputs = None
        self.outputs = None

    def forward(self, inputs):
        self.inputs = np.array(inputs)
        self.outputs = np.array([neuron.forward(self.inputs) for neuron in self.neurons])
        return self.outputs

    def backward(self, output_errors, learning_rate, momentum):
        input_errors = np.zeros(self.input_size)

        for i, neuron in enumerate(self.neurons):
            error = output_errors[i]

            delta = error * neuron.sigmoid_derivative(neuron.output)

            for j in range(len(neuron.weights)):
                input_errors[j] += delta * neuron.weights[j]

            for j in range(len(neuron.weights)):
                weight_update = learning_rate * delta * neuron.inputs[j]
                neuron.weight_momentums[j] = momentum * neuron.weight_momentums[j] + weight_update
                neuron.weights[j] += neuron.weight_momentums[j]

            if neuron.use_bias:
                bias_update = learning_rate * delta
                neuron.bias_momentum = momentum * neuron.bias_momentum + bias_update
                neuron.bias += neuron.bias_momentum

        return input_errors

    def get_weights(self):
        return [neuron.weights for neuron in self.neurons]

    def get_biases(self):
        return [neuron.bias for neuron in self.neurons]

    def get_outputs(self):
        return self.outputs

    def to_dict(self):
        return {
            'input_size': self.input_size,
            'neuron_count': self.neuron_count,
            'neurons': [neuron.to_dict() for neuron in self.neurons]
        }

    @classmethod
    def from_dict(cls, data):
        layer = cls(data['input_size'], data['neuron_count'])
        layer.neurons = [Neuron.from_dict(neuron_data) for neuron_data in data['neurons']]
        return layer