"""
Hebbian learning module for neural networks.

This module implements Hebbian learning, which follows the principle
"neurons that fire together, wire together" to strengthen connections
between co-activated neurons.
"""

import random
import time
from datetime import datetime

class HebbianLearning:
    """
    Implements Hebbian learning for neural networks.
    
    Hebbian learning increases connection strengths between neurons that
    are simultaneously active, creating associative learning in the network.
    This implementation includes methods for tracking learning events and
    modifying the learning rate.
    
    Attributes:
        network: Reference to the neural network
        config: Configuration parameters
        last_learning_time: Timestamp of last learning cycle
        learning_events: History of learning events
        excluded_neurons: List of neurons to exclude from learning
        learning_rate: Current learning rate
    """
    
    def __init__(self, network):
        """
        Initialize Hebbian learning module.
        
        Args:
            network: Neural network to apply learning to
        """
        self.network = network
        self.config = network.config
        self.last_learning_time = time.time()
        self.learning_events = []
        self.excluded_neurons = []
        self.learning_rate = self.config.hebbian.get('base_learning_rate', 0.1)
        
    def perform_hebbian_learning(self):
        """
        Apply Hebbian learning to the network based on current state.
        
        Identifies active neurons and strengthens connections between them.
        Uses thresholding and sampling to control the learning process.
        
        Returns:
            list: Pairs of neurons updated during learning
        """
        current_time = time.time()
        updated_pairs = []
        
        # Debounce - prevent too frequent learning
        min_interval = 5  # 5 second minimum interval
        if current_time - self.last_learning_time < min_interval:
            return updated_pairs
            
        self.last_learning_time = current_time
        
        # Find active neurons
        active_neurons = []
        threshold = self.config.hebbian.get('active_threshold', 50)
        
        for name, value in self.network.state.items():
            if name in self.excluded_neurons:
                continue
                
            # Convert different value types to numeric
            numeric_value = self.network.get_neuron_value(name)
            
            if numeric_value > threshold:
                active_neurons.append(name)
        
        # If less than two neurons are active, no learning occurs
        if len(active_neurons) < 2:
            return updated_pairs
            
        # Sample random pairs to learn (to prevent too many updates)
        sample_size = min(2, len(active_neurons) * (len(active_neurons) - 1) // 2)
        neuron_pairs = [(i, j) for i in range(len(active_neurons)) for j in range(i+1, len(active_neurons))]
        
        if len(neuron_pairs) > sample_size:
            sampled_pairs = random.sample(neuron_pairs, sample_size)
        else:
            sampled_pairs = neuron_pairs
            
        # Update connections for selected pairs
        for i, j in sampled_pairs:
            neuron1 = active_neurons[i]
            neuron2 = active_neurons[j]
            value1 = self.network.get_neuron_value(neuron1)
            value2 = self.network.get_neuron_value(neuron2)
            
            # Only learn if both values are above threshold
            if value1 > threshold and value2 > threshold:
                self._update_connection(neuron1, neuron2, value1, value2)
                updated_pairs.append((neuron1, neuron2))
        
        return updated_pairs
    
    def _update_connection(self, neuron1, neuron2, value1, value2):
        """
        Update the connection weight between two neurons.
        
        Increases the connection weight based on the activation values
        of both neurons, creating or strengthening the associations.
        
        Args:
            neuron1 (str): First neuron name
            neuron2 (str): Second neuron name
            value1 (float): Activation value of first neuron
            value2 (float): Activation value of second neuron
        """
        # Ensure we have connections in both directions
        key_forward = (neuron1, neuron2)
        key_backward = (neuron2, neuron1)
        
        # Create connections if they don't exist
        if key_forward not in self.network.connections:
            self.network.connect(neuron1, neuron2, 0.0)
        
        if key_backward not in self.network.connections:
            self.network.connect(neuron2, neuron1, 0.0)
        
        # Get current connection weights
        forward_conn = self.network.connections[key_forward]
        backward_conn = self.network.connections[key_backward]
        
        prev_weight_forward = forward_conn.get_weight()
        prev_weight_backward = backward_conn.get_weight()
        
        # Calculate normalized values (0-1 range)
        norm1 = value1 / 100.0
        norm2 = value2 / 100.0
        
        # Calculate weight changes based on coactivation
        learning_rate = self.learning_rate
        weight_change = learning_rate * norm1 * norm2
        
        # Apply weight changes
        forward_conn.set_weight(prev_weight_forward + weight_change)
        backward_conn.set_weight(prev_weight_backward + weight_change)
        
        # Log learning event
        event = {
            'timestamp': datetime.now().isoformat(),
            'neuron1': neuron1,
            'neuron2': neuron2,
            'value1': value1,
            'value2': value2,
            'prev_weight_forward': prev_weight_forward,
            'new_weight_forward': forward_conn.get_weight(),
            'prev_weight_backward': prev_weight_backward,
            'new_weight_backward': backward_conn.get_weight(),
            'learning_rate': learning_rate
        }
        
        self.learning_events.append(event)
        
        # Keep event history manageable
        if len(self.learning_events) > 100:
            self.learning_events = self.learning_events[-100:]
    
    def modify_learning_rate(self, factor):
        """
        Modify the learning rate by a factor.
        
        Useful for boosting learning during important events
        or reducing it during less critical periods.
        
        Args:
            factor (float): Multiplier for the learning rate
        
        Returns:
            float: New learning rate
        """
        base_rate = self.config.hebbian.get('base_learning_rate', 0.1)
        self.learning_rate = base_rate * factor
        return self.learning_rate
    
    def get_recent_learning_events(self, count=10):
        """
        Get the most recent learning events.
        
        Provides history data that can be used for analysis
        or visualization of the learning process.
        
        Args:
            count (int, optional): Number of events to return. Defaults to 10.
        
        Returns:
            list: Recent learning events
        """
        return self.learning_events[-count:]
    
    def reset_learning_history(self):
        """
        Clear the learning event history.
        
        Useful for starting a new experiment or reducing memory usage.
        """
        self.learning_events = []