import numpy as np
import json
import random

from Layer import Layer

class MLP:
    def __init__(self, layers, use_bias=True, learning_rate=0.1, momentum=0.0):
        self.layers = []
        self.layer_sizes = layers
        self.use_bias = use_bias
        self.learning_rate = learning_rate
        self.momentum = momentum

        for i in range(len(layers) - 1):
            layer = Layer(layers[i], layers[i + 1], use_bias)
            self.layers.append(layer)

    def forward(self, inputs):
        outputs = np.array(inputs)
        for layer in self.layers:
            outputs = layer.forward(outputs)
        return outputs

    def train(self, inputs, targets, epochs, error_limit, shuffle=True, log_interval=50, error_file="errors.txt"):
        errors = []
        inputs = np.array(inputs)
        targets = np.array(targets)

        with open(error_file, "w") as f:
            f.write("epoch,error\n")

        for epoch in range(epochs):
            epoch_error = 0.0

            indices = list(range(len(inputs)))
            if shuffle:
                random.shuffle(indices)

            for idx in indices:
                input_data = inputs[idx]
                target = targets[idx]

                output = self.forward(input_data)

                output_error = target - output
                sample_error = np.mean(output_error ** 2)
                epoch_error += sample_error

                for layer in reversed(self.layers):
                    output_error = layer.backward(output_error, self.learning_rate, self.momentum)

            avg_error = epoch_error / len(inputs)
            errors.append(avg_error)

            if (epoch + 1) % log_interval == 0 or epoch == 0:
                print(f"Epoka {epoch + 1}/{epochs}: Średni błąd = {avg_error:.6f}")

                with open(error_file, "a") as f:
                    f.write(f"{epoch + 1},{avg_error}\n")

            if avg_error < error_limit:
                print(f"Uzyskano poziam błędu w epoce {epoch + 1}. Uczienie przerwane.")

                with open(error_file, "a") as f:
                    f.write(f"{epoch + 1},{avg_error}\n")
                break

        return errors

    def test(self, inputs, targets, output_file="test_results.txt"):
        inputs = np.array(inputs)
        targets = np.array(targets)
        results = []

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("# Wyniki testu\n\n")

            total_error = 0.0
            for i, (input_data, target) in enumerate(zip(inputs, targets)):
                output = self.forward(input_data)

                output_error = target - output
                sample_error = np.mean(output_error ** 2)
                total_error += sample_error

                f.write(f"## Próba {i + 1}\n")
                f.write(f"Wejście: {input_data.tolist()}\n")
                f.write(f"Cel: {target.tolist()}\n")
                f.write(f"Wyjście: {output.tolist()}\n")
                f.write(f"Błąd wyjścia: {output_error.tolist()}\n")
                f.write(f"Błąd próby: {sample_error}\n\n")

                results.append({
                    'input': input_data.tolist(),
                    'target': target.tolist(),
                    'output': output.tolist(),
                    'error': output_error.tolist(),
                    'sample_error': sample_error
                })

            f.write("### Informacje o warstwach\n")
            for layer_idx, layer in enumerate(self.layers):
                f.write(f"Warstwa {layer_idx + 1} (Wyjścia): {layer.get_outputs().tolist()}\n")
                f.write(f"Warstwa {layer_idx + 1} (Wagi):\n")
                for k, weights in enumerate(layer.get_weights()):
                    f.write(f"  Neuron {k + 1}: {weights}\n")

                f.write(f"Warstwa {layer_idx + 1} (Biasy):\n")
                for k, biases in enumerate(layer.get_biases()):
                    f.write(f"  Neuron {k + 1}: {biases}\n")
            f.write("\n")

            avg_error = total_error / len(inputs)
            f.write(f"# Podsumowanie\n")
            f.write(f"Średni błąd: {avg_error}\n")

        return {
            'results': results,
            'avg_error': avg_error
        }

    def save(self, filename):
        data = {
            'layer_sizes': self.layer_sizes,
            'use_bias': self.use_bias,
            'learning_rate': self.learning_rate,
            'momentum': self.momentum,
            'layers': [layer.to_dict() for layer in self.layers]
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filename):
        with open(filename, 'r') as f:
            data = json.load(f)

        network = cls(
            data['layer_sizes'],
            data['use_bias'],
            data['learning_rate'],
            data['momentum']
        )

        network.layers = [Layer.from_dict(layer_data) for layer_data in data['layers']]

        return network