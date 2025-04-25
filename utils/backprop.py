"""
Backpropagation module for neural networks.

This module provides supervised learning capabilities using the
backpropagation algorithm, which adjusts connection weights
to minimize the difference between actual and expected outputs.
"""

import math
import random
import numpy as np

def sigmoid(x):
    """
    Sigmoid activation function.
    
    Transforms input values into the range (0,1) using the sigmoid function.
    Includes protection against overflow for extreme negative values.
    
    Args:
        x (float): Input value
        
    Returns:
        float: Sigmoid of x (0-1 range)
    """
    try:
        if x < -700:  # Prevent overflow
            return 0
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0 if x < 0 else 1

class BackpropNetwork:
    """
    Backpropagation extension for neural networks.
    
    This class adds supervised learning capabilities to any network
    using the backpropagation algorithm. It organizes neurons into layers,
    performs forward propagation to generate outputs, and backward propagation
    to adjust weights based on error.
    
    Attributes:
        network: The neural network to extend
        learning_rate (float): Learning rate for weight updates
        momentum (float): Momentum coefficient for smoother convergence
        layers (list): List of lists of neuron names, defining network layers
        prev_weight_changes (dict): Previous weight changes for momentum calculation
    """
    
    def __init__(self, network):
        """
        Initialize a backpropagation wrapper for an existing neural network.
        
        Args:
            network: The neural network to add backpropagation capabilities to
        """
        self.network = network
        self.learning_rate = 0.1
        self.momentum = 0.9  # Add momentum for better convergence
        self.layers = []     # List of lists of neuron names
        self.prev_weight_changes = {}  # For momentum
        
    def set_layers(self, layers):
        """
        Define network layers for forward/backward passes.
        
        Organizes neurons into layers for the feedforward architecture.
        The first layer is considered the input layer, the last layer is
        the output layer, and all layers in between are hidden layers.
        
        Args:
            layers (list): List of lists, where each inner list contains neuron names
                          for that layer (e.g., [['input1', 'input2'], ['hidden1'], ['output1']])
        """
        self.layers = layers
        self.prev_weight_changes = {}  # Reset weight changes
        
    def forward_pass(self, inputs):
        """
        Run forward pass through the network.
        
        Takes input values, applies them to the input layer neurons,
        and propagates activation through the network layers.
        
        Args:
            inputs (list): Values to apply to the input layer neurons
            
        Returns:
            list: Output values from the output layer neurons
            
        Raises:
            ValueError: If network layers have not been defined
        """
        if not self.layers:
            raise ValueError("Network layers not defined")
            
        # Set input values
        input_layer = self.layers[0]
        for i, neuron_name in enumerate(input_layer):
            if i < len(inputs):
                self.network.state[neuron_name] = inputs[i]
        
        # Propagate through hidden and output layers
        for layer_idx in range(1, len(self.layers)):
            current_layer = self.layers[layer_idx]
            prev_layer = self.layers[layer_idx-1]
            
            for neuron_name in current_layer:
                # Sum weighted inputs
                weighted_sum = 0
                for prev_neuron in prev_layer:
                    # Check for connection in either direction
                    if (prev_neuron, neuron_name) in self.network.connections:
                        conn = self.network.connections[(prev_neuron, neuron_name)]
                        weight = conn.get_weight()
                        value = self.network.state[prev_neuron]
                        weighted_sum += weight * (value / 100.0)  # Normalize to 0-1
                
                # Apply activation function and scale back to 0-100
                activation = sigmoid(weighted_sum)
                self.network.state[neuron_name] = activation * 100.0
        
        # Return output layer values
        return [self.network.state[n] for n in self.layers[-1]]
    
    def backward_pass(self, targets):
        """
        Perform backward pass with error propagation.
        
        Calculates output errors, backpropagates them through the network,
        and updates connection weights accordingly.
        
        Args:
            targets (list): Target values for the output layer neurons
            
        Returns:
            tuple: (mean squared error, total weight changes)
            
        Raises:
            ValueError: If network layers have not been defined or if targets don't match output layer
        """
        if not self.layers:
            raise ValueError("Network layers not defined")
            
        if len(targets) != len(self.layers[-1]):
            raise ValueError(f"Expected {len(self.layers[-1])} targets, got {len(targets)}")
        
        # Normalize targets to 0-1
        normalized_targets = [t / 100.0 for t in targets]
        
        # Calculate output layer errors
        output_layer = self.layers[-1]
        deltas = {}
        
        for i, neuron_name in enumerate(output_layer):
            # Get normalized output (0-1)
            output = self.network.state[neuron_name] / 100.0
            target = normalized_targets[i]
            
            # Output error * derivative of sigmoid
            error = target - output
            deltas[neuron_name] = error * output * (1 - output)
        
        # Backpropagate errors through hidden layers
        for layer_idx in range(len(self.layers) - 2, -1, -1):
            current_layer = self.layers[layer_idx]
            next_layer = self.layers[layer_idx + 1]
            
            for neuron_name in current_layer:
                error_sum = 0
                
                # Sum weighted deltas from next layer
                for next_neuron in next_layer:
                    if (neuron_name, next_neuron) in self.network.connections:
                        connection = self.network.connections[(neuron_name, next_neuron)]
                        weight = connection.get_weight()
                        error_sum += weight * deltas[next_neuron]
                
                # Get normalized output and calculate delta
                output = self.network.state[neuron_name] / 100.0
                deltas[neuron_name] = error_sum * output * (1 - output)
        
        # Update weights
        total_changes = 0
        for layer_idx in range(len(self.layers) - 1):
            current_layer = self.layers[layer_idx]
            next_layer = self.layers[layer_idx + 1]
            
            for neuron_name in current_layer:
                output = self.network.state[neuron_name] / 100.0
                
                for next_neuron in next_layer:
                    conn_key = (neuron_name, next_neuron)
                    if conn_key in self.network.connections:
                        connection = self.network.connections[conn_key]
                        
                        # Calculate weight delta
                        delta_w = self.learning_rate * deltas[next_neuron] * output
                        
                        # Add momentum term if available
                        if conn_key in self.prev_weight_changes:
                            delta_w += self.momentum * self.prev_weight_changes[conn_key]
                        
                        # Store for next iteration
                        self.prev_weight_changes[conn_key] = delta_w
                        
                        # Update weight
                        old_weight = connection.get_weight()
                        connection.set_weight(old_weight + delta_w)
                        
                        total_changes += abs(delta_w)
        
        # Calculate error for reporting
        mse = 0
        for i, neuron_name in enumerate(output_layer):
            output = self.network.state[neuron_name] / 100.0
            target = normalized_targets[i]
            mse += (target - output) ** 2
        
        mse /= len(output_layer)
        
        return mse, total_changes
        
    def train(self, training_data, epochs=100, target_error=0.01, callback=None):
        """
        Train the network on a dataset.
        
        Performs multiple epochs of training, each consisting of forward and backward passes
        for each training example. The order of training examples is randomized for each epoch.
        
        Args:
            training_data (list): List of (inputs, targets) tuples for training
            epochs (int, optional): Maximum number of training epochs. Defaults to 100.
            target_error (float, optional): Error threshold to stop training. Defaults to 0.01.
            callback (callable, optional): Function called after each epoch with 
                                          (epoch, error) parameters. If it returns 
                                          False, training is stopped. Defaults to None.
            
        Returns:
            list: Error values for each epoch
        """
        errors = []
        
        for epoch in range(epochs):
            epoch_error = 0
            
            # Randomize training data order
            random.shuffle(training_data)
            
            for inputs, targets in training_data:
                # Forward pass
                self.forward_pass(inputs)
                
                # Backward pass
                error, _ = self.backward_pass(targets)
                epoch_error += error
            
            # Average error for epoch
            avg_error = epoch_error / len(training_data)
            errors.append(avg_error)
            
            # Call progress callback if provided
            if callback and callable(callback):
                if not callback(epoch, avg_error):
                    break  # Callback returned False, stop training
            
            # Check for target error
            if avg_error <= target_error:
                break
                
        return errors
    
    def test(self, test_data):
        """
        Test network on a dataset and return accuracy.
        
        For classification tasks, compares the maximum output value to the
        maximum target value to determine correctness.
        
        Args:
            test_data (list): List of (inputs, targets) tuples for testing
            
        Returns:
            float: Accuracy as a proportion of correct predictions (0.0-1.0)
        """
        if not test_data:
            return 0
            
        correct = 0
        for inputs, targets in test_data:
            outputs = self.forward_pass(inputs)
            
            # For classification tasks, find max output
            max_idx = outputs.index(max(outputs))
            target_idx = targets.index(max(targets)) if isinstance(targets, list) else 0
            
            if max_idx == target_idx:
                correct += 1
                
        return correct / len(test_data)