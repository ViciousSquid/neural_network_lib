"""
XOR problem example using backpropagation.

This example demonstrates how to use the backpropagation module
to train a neural network to solve the XOR problem.
"""

import numpy as np
import matplotlib.pyplot as plt
from neural_network_lib.core import Network, Config
from neural_network_lib.learning import BackpropNetwork

def main():
    # Create a neural network for XOR problem
    config = Config()
    network = Network(config)
    
    # Create neurons for input, hidden, and output layers
    network.add_neuron("input1", position=(100, 100))
    network.add_neuron("input2", position=(100, 200))
    network.add_neuron("hidden1", position=(250, 100))
    network.add_neuron("hidden2", position=(250, 200))
    network.add_neuron("output", position=(400, 150))
    
    # Create initial random connections
    for input_name in ["input1", "input2"]:
        for hidden_name in ["hidden1", "hidden2"]:
            weight = np.random.uniform(-0.5, 0.5)
            network.connect(input_name, hidden_name, weight)
    
    for hidden_name in ["hidden1", "hidden2"]:
        weight = np.random.uniform(-0.5, 0.5)
        network.connect(hidden_name, "output", weight)
    
    # Create backpropagation network
    backprop = BackpropNetwork(network)
    backprop.set_layers([
        ["input1", "input2"],      # Input layer
        ["hidden1", "hidden2"],    # Hidden layer
        ["output"]                 # Output layer
    ])
    
    # Configure backprop parameters
    backprop.learning_rate = 0.2
    backprop.momentum = 0.9
    
    # Define XOR training data: (inputs, expected output)
    training_data = [
        ([0, 0], [0]),    # 0 XOR 0 = 0
        ([0, 100], [100]), # 0 XOR 1 = 1
        ([100, 0], [100]), # 1 XOR 0 = 1
        ([100, 100], [0])  # 1 XOR 1 = 0
    ]
    
    # Train the network
    print("Training network on the XOR problem...")
    epochs = 1000
    errors = []
    
    # Progress callback
    def progress_callback(epoch, error):
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: error = {error:.6f}")
        errors.append(error)
        return True  # Continue training
    
    # Train the network
    backprop.train(training_data, epochs=epochs, target_error=0.01, callback=progress_callback)
    
    # Test the trained network
    print("\nTesting trained network:")
    correct = 0
    for inputs, expected in training_data:
        outputs = backprop.forward_pass(inputs)
        predicted = 1 if outputs[0] > 50 else 0
        expected_val = 1 if expected[0] > 50 else 0
        
        print(f"Input: {[1 if x > 50 else 0 for x in inputs]}, "
              f"Output: {outputs[0]:.1f}, Predicted: {predicted}, "
              f"Expected: {expected_val}")
        
        if (predicted == expected_val):
            correct += 1
    
    accuracy = correct / len(training_data)
    print(f"\nAccuracy: {accuracy * 100:.1f}%")
    
    # Plot error over epochs
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(errors)), errors)
    plt.title('XOR Training Error')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.grid(True)
    plt.yscale('log')  # Log scale to see small errors better
    plt.savefig('xor_training_error.png')
    plt.show()
    
    print("\nFinal weights:")
    for (src, tgt), conn in network.connections.items():
        print(f"{src} → {tgt}: {conn.get_weight():.4f}")
    
    # Show strongest connections
    print("\nStrongest connections:")
    for src, tgt, weight in network.get_strongest_connections():
        print(f"{src} → {tgt}: {weight:.4f}")

if __name__ == "__main__":
    main()
