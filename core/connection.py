"""
Connection module for neural networks.

This module defines the Connection class, which represents a weighted
link between two neurons in the network.
"""

import time

class Connection:
    """
    Represents a weighted connection between two neurons.
    
    Connections are directional links between neurons that determine
    how activation flows through the network. Each connection has a
    weight that can be positive (excitatory) or negative (inhibitory).
    
    Attributes:
        source (str): Source neuron name
        target (str): Target neuron name
        weight (float): Connection weight (-1.0 to 1.0)
        last_update (float): Timestamp of last weight update
        creation_time (float): Timestamp of connection creation
        weight_history (list): History of weight changes with timestamps
        bidirectional (bool): Whether this is part of a bidirectional connection
    """
    
    def __init__(self, source, target, weight=0, bidirectional=False):
        """
        Initialize a connection between two neurons.
        
        Args:
            source (str): Source neuron name
            target (str): Target neuron name
            weight (float, optional): Initial connection weight. Defaults to 0.
            bidirectional (bool, optional): If True, this is part of a bidirectional pair. Defaults to False.
        """
        self.source = source
        self.target = target
        self._weight = 0  # Use property to enforce bounds
        self.weight_history = []  # Initialize BEFORE calling set_weight
        self.last_update = time.time()
        self.creation_time = time.time()
        self.set_weight(weight)  # Now safe to call
        self.bidirectional = bidirectional
        
    def get_weight(self):
        """
        Get the current connection weight.
        
        Returns:
            float: Current weight value
        """
        return self._weight
        
    def set_weight(self, weight):
        """
        Set the connection weight, clamping to valid range.
        
        Args:
            weight (float): New weight value
        
        Returns:
            float: Actual weight value after clamping
        """
        # Clamp weight to [-1, 1]
        self._weight = max(-1.0, min(1.0, weight))
        self.last_update = time.time()
        self.weight_history.append((time.time(), self._weight))
        
        # Keep history manageable
        if len(self.weight_history) > 100:
            self.weight_history = self.weight_history[-100:]
            
        return self._weight
        
    def apply_decay(self, decay_factor=0.01):
        """
        Apply decay to the connection weight.
        
        Reduces the absolute weight value over time, preventing
        runaway weight growth during learning.
        
        Args:
            decay_factor (float, optional): Decay factor (0-1). Defaults to 0.01.
            
        Returns:
            float: New weight after decay
        """
        return self.set_weight(self._weight * (1 - decay_factor))
        
    def is_excitatory(self):
        """
        Check if this is an excitatory connection (positive weight).
        
        Returns:
            bool: True if weight is positive, False otherwise
        """
        return self._weight > 0
        
    def is_inhibitory(self):
        """
        Check if this is an inhibitory connection (negative weight).
        
        Returns:
            bool: True if weight is negative, False otherwise
        """
        return self._weight < 0
        
    def __repr__(self):
        """
        String representation of the connection.
        
        Returns:
            str: A human-readable representation of the connection
        """
        return f"Connection({self.source} → {self.target}, weight={self._weight:.3f})"