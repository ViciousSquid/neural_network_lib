"""
Neuron module for neural networks.

This module defines the Neuron class, which represents a single
computational unit in a neural network.
"""

class Neuron:
    """
    Represents a single neuron in the neural network.
    
    A neuron serves as a node in the network graph, with properties including
    position, type, and activation history. Neurons are connected to other
    neurons via weighted connections.
    
    Attributes:
        name (str): Unique identifier for the neuron
        position (tuple): 2D position coordinates (x, y) for visualization
        type (str): Type of neuron (default, novelty, stress, reward, etc.)
        attributes (dict): Optional additional attributes for the neuron
        activity_history (list): Recent activation values
    """
    
    def __init__(self, name, position=None, neuron_type="default", attributes=None):
        """
        Initialize a neuron with a name and optional position.
        
        Args:
            name (str): Unique identifier for this neuron
            position (tuple, optional): (x, y) position for visualization. Defaults to (0, 0).
            neuron_type (str, optional): Type of neuron. Defaults to "default".
            attributes (dict, optional): Additional attributes. Defaults to None.
        """
        self.name = name
        self.position = position or (0, 0)
        self.type = neuron_type
        self.attributes = attributes or {}
        self.activity_history = []  # For tracking recent activation
        
    def get_position(self):
        """
        Return the neuron's position.
        
        Returns:
            tuple: (x, y) coordinates of the neuron
        """
        return self.position
        
    def set_position(self, x, y):
        """
        Set the neuron's position.
        
        Used primarily for visualization and spatial organization.
        
        Args:
            x (float): X coordinate
            y (float): Y coordinate
        """
        self.position = (x, y)
        
    def record_activity(self, value, max_history=10):
        """
        Record neuron activity for tracking over time.
        
        Maintains a rolling history of recent activation values,
        which can be used for analysis and visualization.
        
        Args:
            value (float): Activity value to record
            max_history (int, optional): Maximum history length. Defaults to 10.
        """
        self.activity_history.append(value)
        if len(self.activity_history) > max_history:
            self.activity_history.pop(0)
            
    def get_mean_activity(self):
        """
        Calculate mean activity from recent history.
        
        Returns:
            float: Average of recent activation values, or 0 if no history
        """
        if not self.activity_history:
            return 0
        return sum(self.activity_history) / len(self.activity_history)
            
    def __repr__(self):
        """
        String representation of the neuron.
        
        Returns:
            str: A human-readable representation of the neuron
        """
        return f"Neuron('{self.name}', pos={self.position}, type='{self.type}')"