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

# def plot_activation_functions():
#     x = np.linspace(-10, 10, 100)
#     plt.figure(figsize=(12, 8))
#     for i, (func, title) in enumerate(zip([sigmoid, relu, leaky_relu, tanh], 
#                                           ["Sigmoid", "ReLU", "Leaky ReLU", "Tanh"])):
#         plt.subplot(2, 2, i+1)
#         plt.plot(x, func(x), label=title)
#         plt.title(title + " Activation Function")
#         plt.xlabel("Input")
#         plt.ylabel("Output")
#         plt.grid(True)
#         plt.legend()

#     plt.tight_layout()
#     plt.show()
# plot_activation_functions()

def print_activation_outputs_and_plot(random_values):
      outputs = {
       "ReLU": relu(random_values),
       "Leaky ReLU": leaky_relu(random_values),
       "Tanh": tanh(random_values),
   }
  
      for name, output in outputs.items():
       print(f"{name} Output:", output)


      plt.figure(figsize=(12, 4))
      for i, (name, output) in enumerate(outputs.items(), start=1):
       plt.subplot(1, len(outputs), i)
       plt.stem(random_values, output)
       plt.title(name + " Output")
       plt.xlabel("Input")
       plt.ylabel("Output")
       plt.grid(True)


      plt.tight_layout()
      plt.show()


random_values = np.array([-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6])


print_activation_outputs_and_plot(random_values)

