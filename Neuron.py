import random
import numpy as np

class Neuron:
    def __init__(self, input_size, use_bias=True):
        self.weights = [random.uniform(-0.5, 0.5) for _ in range(input_size)]
        self.bias = random.uniform(-0.5, 0.5) if use_bias else 0.0
        self.use_bias = use_bias
        self.inputs = []
        self.output = 0.0
        self.weight_momentums = [0.0 for _ in range(input_size)]
        self.bias_momentum = 0.0

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, inputs):
        self.inputs = np.array(inputs)
        sum = 0
        for i in range(len(inputs)):
            sum += inputs[i] * self.weights[i]

        if self.use_bias:
            sum += self.bias

        self.output = self.sigmoid(sum)
        return self.output

    def to_dict(self):
        return {
            'weights': self.weights,
            'bias': self.bias,
            'use_bias': self.use_bias,
            'weight_momentums': self.weight_momentums,
            'bias_momentum': self.bias_momentum
        }

    @classmethod
    def from_dict(cls, data):
        neuron = cls(len(data['weights']), data['use_bias'])
        neuron.weights = data['weights']
        neuron.bias = data['bias']
        neuron.weight_momentums = data.get('weight_momentums', [0.0] * len(data['weights']))
        neuron.bias_momentum = data.get('bias_momentum', 0.0)
        return neuron

