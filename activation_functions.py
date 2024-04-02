import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, x * alpha)

def tanh(x):
    return np.tanh(x)

def plot_activation_functions():
    x = np.linspace(-10, 10, 100)
    plt.figure(figsize=(12, 8))
    for i, (func, title) in enumerate(zip([sigmoid, relu, leaky_relu, tanh], 
                                          ["Sigmoid", "ReLU", "Leaky ReLU", "Tanh"])):
        plt.subplot(2, 2, i+1)
        plt.plot(x, func(x), label=title)
        plt.title(title + " Activation Function")
        plt.xlabel("Input")
        plt.ylabel("Output")
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.show()
plot_activation_functions()
