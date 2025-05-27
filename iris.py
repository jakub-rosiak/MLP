from MLP import MLP
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

def evaluate_results(results):
    y_true = []
    y_pred = []

    for sample in results:
        y_true.append(np.argmax(sample['target']))
        y_pred.append(np.argmax(sample['output']))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    total_correct = np.sum(y_true == y_pred)
    total_samples = len(y_true)

    print(f"Liczba poprawnie sklasyfikowanych obiektów: {total_correct} / {total_samples}")

    unique_classes = np.unique(y_true)
    for cls in unique_classes:
        cls_correct = np.sum((y_true == cls) & (y_pred == cls))
        cls_total = np.sum(y_true == cls)
        print(f"Klasa {cls}: {cls_correct} poprawnie sklasyfikowanych na {cls_total} przykładów")

    cm = confusion_matrix(y_true, y_pred)
    print("\nMacierz pomyłek:")
    for row in cm:
        print(" ".join(map(str, row)))

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, zero_division=0)
    print("\nPrecision, Recall, F1-score dla klas:")
    for idx, cls in enumerate(unique_classes):
        print(f"Klasa {cls}: Precision={precision[idx]:.3f}, Recall={recall[idx]:.3f}, F1-score={f1[idx]:.3f}")

    return {
        "total_correct": total_correct,
        "total_samples": total_samples,
        "per_class_correct": {cls: np.sum((y_true == cls) & (y_pred == cls)) for cls in unique_classes},
        "per_class_total": {cls: np.sum(y_true == cls) for cls in unique_classes},
        "confusion_matrix": cm,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

def save_results_to_file(results_dict, filename="results.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"Liczba poprawnie sklasyfikowanych obiektów: {results_dict['total_correct']} / {results_dict['total_samples']}\n\n")

        f.write("Dokładność dla poszczególnych klas:\n")
        for cls in sorted(results_dict['per_class_total'].keys()):
            cls_correct = results_dict['per_class_correct'][cls]
            cls_total = results_dict['per_class_total'][cls]
            f.write(f"Klasa {cls}: {cls_correct} poprawnie sklasyfikowanych na {cls_total} przykładów\n")

        f.write("\nMacierz pomyłek:\n")
        cm = results_dict["confusion_matrix"]
        for row in cm:
            f.write(" ".join(map(str, row)) + "\n")

        f.write("\nPrecision, Recall, F1-score dla klas:\n")
        for idx, cls in enumerate(sorted(results_dict['per_class_total'].keys())):
            precision = results_dict['precision'][idx]
            recall = results_dict['recall'][idx]
            f1 = results_dict['f1_score'][idx]
            f.write(f"Klasa {cls}: Precision={precision:.3f}, Recall={recall:.3f}, F1-score={f1:.3f}\n")

def load_iris_data():
    try:
        data = pd.read_csv("data/iris.data", header=None, names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])
        print("Dane załadowane poprawnie.")

        lb = LabelBinarizer()
        data['class'] = lb.fit_transform(data['class']).tolist()
        return data
    except FileNotFoundError:
        print("Plik nie został znaleziony. Sprawdź ścieżkę pliku.")
    except Exception as e:
        print(f"Wystąpił błąd: {e}")


def get_layers():
    layers = [4]
    while True:
        choice = input(f"Podaj liczbę neuronów dla warstwy ukrytej {len(layers)}, lub 'q' aby zakończyć: ")
        if choice.lower() == 'q':
            layers.append(3)
            return layers
        try:
            value = int(choice)
            if value < 1:
                print("Liczba neuronów musi być większa niż 0.")
                continue
            layers.append(value)
        except ValueError:
            print("Nieprawidłowa wartość, spróbuj ponownie.")

def use_bias():
    while True:
        choice = input("Czy chcesz użyć biasu w warstwach? (tak/nie): ").strip().lower()
        if choice in ['tak', 'nie']:
            return choice == 'tak'
        else:
            print("Nieprawidłowa odpowiedź. Wpisz 'tak' lub 'nie'.")

def get_epoch():
    while True:
        inp = input("Podaj liczbę epok (np. 100): ")
        try:
            epochs = int(inp)
            if epochs > 0:
                return epochs
            else:
                print("Liczba epok musi być dodatnia.")
        except ValueError:
            print("Nieprawidłowa wartość, podaj liczbę całkowitą.")

def get_error_limit():
    while True:
        inp = input("Podaj limit błędu (np. 0.01): ")
        try:
            error_limit = float(inp)
            if error_limit > 0:
                return error_limit
            else:
                print("Limit błędu musi być dodatni.")
        except ValueError:
            print("Nieprawidłowa wartość, podaj liczbę zmiennoprzecinkową.")

def get_learning_rate():
    while True:
        try:
            lr = float(input("Podaj learning rate (współczynnik uczenia): "))
            if lr > 0:
                return lr
            else:
                print("Learning rate musi być większe od zera.")
        except ValueError:
            print("Nieprawidłowa wartość. Podaj liczbę.")

def get_momentum():
    while True:
        try:
            momentum = float(input("Podaj momentum: "))
            if 0 <= momentum <= 1:
                return momentum
            else:
                print("Momentum musi być w zakresie 0-1.")
        except ValueError:
            print("Nieprawidłowa wartość. Podaj liczbę.")

def save_network(mlp):
    while True:
        inp = input("Czy chcesz zapisać sieć? (tak/nie): ").strip().lower()
        if inp in ['tak', 'nie']:
            if inp == 'tak':
                file = input("Podaj plik do zapisu: ")
                try:
                    mlp.save(f"./results/iris/{file}")
                    print(f"Sieć zapisana w pliku {file}.")
                except Exception as e:
                    print(f"Wystąpił błąd podczas zapisu: {e}")
            else:
                print("Sieć nie została zapisana.")
            return None
        else:
            print("Nieprawidłowa odpowiedź. Wpisz 'tak' lub 'nie'.")

def load_network():
    while True:
        inp = input("Czy chcesz wczytać sieć? (tak/nie): ").strip().lower()
        if inp in ['tak', 'nie']:
            if inp == 'tak':
                file = input("Podaj plik do odczytu: ")
                try:
                    mlp = MLP.load(f"./results/iris/{file}")
                    print(f"Sieć wczytana z pliku {file}.")
                    return mlp
                except Exception as e:
                    print(f"Wystąpił błąd podczas odczytu: {e}")
            else:
                print("Sieć nie została wczytana.")
            return None
        else:
            print("Nieprawidłowa odpowiedź. Wpisz 'tak' lub 'nie'.")



def main():
    loaded = True
    x_train, x_test, y_train, y_test = None, None, None, None
    mlp = load_network()
    if mlp is None:
        loaded = False
        layers = get_layers()
        bias = use_bias()
        learning_rate = get_learning_rate()
        momentum = get_momentum()
        mlp = MLP(layers, bias, learning_rate, momentum)

        iris = load_iris_data()
        x = iris.iloc[:, :-1].values
        y = np.array(iris['class'].tolist())

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

        epochs = get_epoch()
        error_limit = get_error_limit()
        mlp.train(x_train, y_train, epochs, error_limit, error_file="./results/iris/errors.txt")

    iris = load_iris_data()
    x = iris.iloc[:, :-1].values
    y = np.array(iris['class'].tolist())

    if x_train is None or x_test is None or y_test is None or y_train is None:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

    results = mlp.test(x_test, y_test, f"./results/iris/iris.txt")

    metrics = evaluate_results(results['results'])

    save_results_to_file(metrics, "./results/iris/metrics.txt")

    if not loaded:
        save_network(mlp)

if __name__ == '__main__':
    main()