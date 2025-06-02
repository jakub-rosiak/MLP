from MLP import MLP
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def generate_plot(error_file, save_path):
    df = pd.read_csv(error_file)

    if df.empty or 'epoch' not in df.columns or 'error' not in df.columns:
        print(f"Uszkodzony plik: {error_file}")
        return

    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="epoch", y="error", marker='o', color='steelblue')
    plt.title("Bład względem epok")
    plt.xlabel("Epoka")
    plt.ylabel("Błąd")
    plt.ylim(bottom=0)
    plt.tight_layout()

    plt.savefig(save_path)
    print(f"Wykres zapisany do: {save_path}")

def train_autoencoder(bias, learning_rate, momentum, epochs, error_limit, filename, error_file, plot_path):

    patterns = [
        ([1, 0, 0, 0], [1, 0, 0, 0]),
        ([0, 1, 0, 0], [0, 1, 0, 0]),
        ([0, 0, 1, 0], [0, 0, 1, 0]),
        ([0, 0, 0, 1], [0, 0, 0, 1])
    ]

    layers = [4, 2, 4]

    mlp = MLP(layers, bias, learning_rate, momentum)

    x = np.array([p[0] for p in patterns])
    y = np.array([p[1] for p in patterns])

    print("Rozpoczynanie nauki autoenkodera...")
    mlp.train(x, y, epochs, error_limit, error_file=error_file, log_interval=10)

    print("Testowanie autoenkodera...")
    results = mlp.test(x, y, filename)

    for i, result in enumerate(results['results']):
        print(f"Wzorzec wejściowy: {patterns[i][0]}")
        print(f"Wzorzec oczekiwany: {patterns[i][1]}")
        rounded_output = [f"{x:.6f}" for x in result['output']]
        print(f"Odpowiedź sieci: {rounded_output}")

        print("-" * 50)

    generate_plot(error_file, plot_path)
    return results

def main():
    results1 = train_autoencoder(False, 0.6, 0.0, 1000, 0.001, "./results/autoencoder/autoencoder_no_bias.txt", "./results/autoencoder/autoencoder_no_bias_errors.txt", "./results/autoencoder/plots/autoencoder_no_bias_plot.png")
    results2 = train_autoencoder(True, 0.6, 0.0, 1000, 0.001, "./results/autoencoder/autoencoder_bias.txt", "./results/autoencoder/autoencoder_bias_errors.txt", "./results/autoencoder/plots/autoencoder_bias_plot.png")

    training_values = [
        (0.9, 0.0),
        (0.6, 0.0),
        (0.2, 0.0),
        (0.9, 0.6),
        (0.2, 0.9)
    ]
    bias = True if results1['avg_error'] > results2['avg_error'] else False
    for i, value in enumerate(training_values):
        train_autoencoder(bias, value[0], value[1], 500, 0.0001, f"./results/autoencoder/autoencoder_{i}.txt", f"./results/autoencoder/autoencoder_{i}_errors.txt", f"./results/autoencoder/plots/autoencoder_{i}_plot.png")

if __name__ == "__main__":
    main()