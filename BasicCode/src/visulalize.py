# Visualization function
import matplotlib.pyplot as plt

def plot_metrics(rounds, loss, accuracy):
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(rounds, loss, marker='o', color='b', label='Loss')
    plt.title('Loss Over Rounds')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(rounds, accuracy, marker='o', color='r', label='Accuracy')
    plt.title('Accuracy Over Rounds')
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()
