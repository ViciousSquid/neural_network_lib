"""
Neural network core module.

This module defines the Network class, which is the central component
of the neural network library, managing neurons, connections, and state.
"""

import time
import random
import os
import json
from .neuron import Neuron
from .connection import Connection
from .config import Config

class Network:
    """
    Neural network with neurons, connections, and state.
    
    The Network class is the central component of the library, providing
    methods for creating, connecting, and updating neurons, as well as
    performing learning and neurogenesis. It maintains the current activation
    state of all neurons and manages all connections between them.
    
    Attributes:
        config (Config): Configuration parameters
        neurons (dict): Map of neuron names to Neuron objects
        connections (dict): Map of (source, target) tuples to Connection objects
        state (dict): Current activation state of neurons
        creation_time (float): Timestamp of network creation
        last_update_time (float): Timestamp of last state update
        update_count (int): Number of state updates since creation
        excluded_neurons (list): Neurons to exclude from learning
        learning: HebbianLearning component (initialized later)
        neurogenesis: Neurogenesis component (initialized later)
    """
    
    def __init__(self, config=None):
        """
        Initialize neural network.
        
        Args:
            config (Config, optional): Configuration parameters. Defaults to None.
        """
        self.config = config or Config()
        self.neurons = {}  # name -> Neuron
        self.connections = {}  # (source, target) -> Connection
        self.state = {}  # name -> state value (activation)
        
        # Stats and tracking
        self.creation_time = time.time()
        self.last_update_time = time.time()
        self.update_count = 0
        self.excluded_neurons = []  # Neurons to exclude from learning
        
        # Will be initialized later to avoid circular imports
        self.learning = None  
        self.neurogenesis = None
        
    def add_neuron(self, name, initial_state=0, position=None, neuron_type="default", attributes=None):
        """
        Add a neuron to the network.
        
        Creates a new neuron and adds it to the network with the specified
        properties and initial activation state.
        
        Args:
            name (str): Unique neuron name
            initial_state (float, optional): Initial activation state. Defaults to 0.
            position (tuple, optional): (x,y) coordinates. Defaults to None.
            neuron_type (str, optional): Type of neuron. Defaults to "default".
            attributes (dict, optional): Additional attributes. Defaults to None.
            
        Returns:
            Neuron: The created neuron
            
        Raises:
            ValueError: If a neuron with the given name already exists
        """
        if name in self.neurons:
            raise ValueError(f"Neuron '{name}' already exists")
            
        neuron = Neuron(name, position, neuron_type, attributes)
        self.neurons[name] = neuron
        self.state[name] = initial_state
        return neuron
        
    def connect(self, source, target, weight=0, bidirectional=False):
        """
        Create a connection between neurons.
        
        Creates a weighted connection from the source neuron to the target neuron.
        If bidirectional is True, also creates a connection in the reverse direction.
        
        Args:
            source (str): Source neuron name
            target (str): Target neuron name
            weight (float, optional): Connection weight. Defaults to 0.
            bidirectional (bool, optional): If True, create connections in both directions. Defaults to False.
            
        Returns:
            Connection: The created connection
            
        Raises:
            ValueError: If either the source or target neuron does not exist
        """
        # Validate neurons exist
        if source not in self.neurons:
            raise ValueError(f"Source neuron '{source}' does not exist")
        if target not in self.neurons:
            raise ValueError(f"Target neuron '{target}' does not exist")
            
        # Create connection
        connection = Connection(source, target, weight, bidirectional)
        self.connections[(source, target)] = connection
        
        # Create reverse connection if bidirectional
        if bidirectional and (target, source) not in self.connections:
            self.connections[(target, source)] = Connection(target, source, weight, True)
            
        return connection
                
    def update_state(self, new_state):
        """
        Update neuron states with new values.
        
        Updates the activation levels of neurons based on the provided dictionary
        and records the activations in neuron history.
        
        Args:
            new_state (dict): Dict of neuron names to new state values
            
        Returns:
            dict: The updated state
        """
        # Update neuron states
        for name, value in new_state.items():
            if name in self.state:
                self.state[name] = value
                
                # Record activation in neuron history if it exists
                if name in self.neurons:
                    self.neurons[name].record_activity(self.get_neuron_value(name))
        
        self.last_update_time = time.time()
        self.update_count += 1
        
        return self.state
    
    def propagate_activation(self, steps=1):
        """
        Propagate activation through the network for a number of steps.
        
        Simulates signal flow through the network by updating each neuron's
        activation based on weighted inputs from connected neurons.
        
        Args:
            steps (int, optional): Number of propagation steps. Defaults to 1.
            
        Returns:
            dict: The updated state after propagation
        """
        for _ in range(steps):
            new_state = self.state.copy()
            
            # For each neuron, compute incoming activation
            for target in self.neurons:
                incoming = 0
                count = 0
                
                # Sum weighted inputs from connected neurons
                for source in self.neurons:
                    if (source, target) in self.connections:
                        connection = self.connections[(source, target)]
                        incoming += self.get_neuron_value(source) * connection.get_weight()
                        count += 1
                
                # Update target activation if it has connections
                if count > 0:
                    # Normalize by connection count and apply sigmoid-like scaling
                    normalized = incoming / count
                    current = self.get_neuron_value(target)
                    # Blend current with normalized incoming (70% current, 30% incoming)
                    new_state[target] = current * 0.7 + normalized * 0.3
                    
                    # Ensure value stays in bounds
                    new_state[target] = max(0, min(100, new_state[target]))
            
            # Apply the updates
            self.state = new_state
        
        return self.state
    
    def get_neuron_value(self, neuron_name):
        """
        Convert neuron state to numerical value for learning.
        
        Provides a consistent way to get a numerical value for any neuron,
        regardless of how the state is represented.
        
        Args:
            neuron_name (str): Neuron name
            
        Returns:
            float: Numerical value of the neuron's state
        """
        value = self.state.get(neuron_name, 0)
        
        if isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, bool):
            return 100.0 if value else 0.0
        elif isinstance(value, str):
            return 75.0  # Default for string states
        return 0.0
    
    def initialize_learning(self):
        """
        Initialize learning components.
        
        Sets up the Hebbian learning and neurogenesis components.
        This is called separately from __init__ to avoid circular imports.
        """
        # Import here to avoid circular dependencies
        from ..learning.hebbian import HebbianLearning
        from ..learning.neurogenesis import Neurogenesis
        
        self.learning = HebbianLearning(self)
        self.neurogenesis = Neurogenesis(self)
    
    def perform_learning(self):
        """
        Perform Hebbian learning on the network.
        
        Strengthens connections between co-activated neurons according to 
        Hebbian learning principles.
        
        Returns:
            list: Pairs of neurons that were updated
            
        Raises:
            RuntimeError: If learning component has not been initialized
        """
        if self.learning is None:
            self.initialize_learning()
        
        return self.learning.perform_hebbian_learning()
    
    def check_neurogenesis(self, state):
        """
        Check if neurogenesis should occur.
        
        Evaluates the current state against neurogenesis triggers and
        creates new neurons if conditions are met.
        
        Args:
            state (dict): Input state with potential neurogenesis triggers
            
        Returns:
            bool: True if neurogenesis occurred
            
        Raises:
            RuntimeError: If neurogenesis component has not been initialized
        """
        if self.neurogenesis is None:
            self.initialize_learning()
            
        return self.neurogenesis.check_neurogenesis(state)
    
    def get_connection_strength(self, source, target):
        """
        Get the connection strength between two neurons.
        
        Args:
            source (str): Source neuron name
            target (str): Target neuron name
            
        Returns:
            float: Connection weight or 0 if no connection exists
        """
        if (source, target) in self.connections:
            return self.connections[(source, target)].get_weight()
        return 0
    
    def get_strongest_connections(self, count=5):
        """
        Get the strongest connections in the network.
        
        Returns a sorted list of the strongest connections by absolute weight.
        
        Args:
            count (int, optional): Number of connections to return. Defaults to 5.
            
        Returns:
            list: List of (source, target, weight) tuples, sorted by absolute weight
        """
        return sorted(
            [(src, tgt, conn.get_weight()) for (src, tgt), conn in self.connections.items()],
            key=lambda x: abs(x[2]),
            reverse=True
        )[:count]
    
    def get_network_statistics(self):
        """
        Calculate network statistics.
        
        Computes various statistics about the network structure, weights,
        and activation patterns.
        
        Returns:
            dict: Statistics about the network
        """
        if not self.connections:
            return {
                "neurons": len(self.neurons),
                "connections": 0,
                "avg_weight": 0,
                "positive_ratio": 0,
                "negative_ratio": 0,
                "network_age": time.time() - self.creation_time,
                "update_count": self.update_count
            }
            
        total_weight = sum(abs(conn.get_weight()) for conn in self.connections.values())
        avg_weight = total_weight / len(self.connections)
        
        positive = sum(1 for conn in self.connections.values() if conn.get_weight() > 0)
        negative = sum(1 for conn in self.connections.values() if conn.get_weight() < 0)
        
        return {
            "neurons": len(self.neurons),
            "connections": len(self.connections),
            "avg_weight": avg_weight,
            "positive_ratio": positive / len(self.connections) if self.connections else 0,
            "negative_ratio": negative / len(self.connections) if self.connections else 0,
            "network_age": time.time() - self.creation_time,
            "update_count": self.update_count
        }
    
    def apply_weight_decay(self):
        """
        Apply decay to all weights to prevent runaway strengthening.
        
        Reduces the absolute magnitude of all connection weights by a small amount,
        helping to prevent weights from growing too large during learning.
        
        Returns:
            int: Number of connections decayed
        """
        decay_factor = self.config.hebbian.get('weight_decay', 0.01)
        
        count = 0
        for connection in self.connections.values():
            old_weight = connection.get_weight()
            new_weight = connection.apply_decay(decay_factor)
            
            if abs(new_weight - old_weight) > 0.0001:
                count += 1
                
        return count
    
    def save(self, filepath):
        """
        Save the network to a file.
        
        Serializes the entire network, including neurons, connections, state,
        and configuration to a JSON file.
        
        Args:
            filepath (str): File path to save to
            
        Returns:
            bool: True if successful
        """
        try:
            data = {
                "neurons": {
                    name: {
                        "position": neuron.get_position(),
                        "type": neuron.type,
                        "attributes": neuron.attributes
                    } for name, neuron in self.neurons.items()
                },
                "connections": {
                    f"{src}_{tgt}": {
                        "weight": conn.get_weight(),
                        "creation_time": conn.creation_time
                    } for (src, tgt), conn in self.connections.items()
                },
                "state": self.state,
                "config": {
                    "hebbian": self.config.hebbian,
                    "neurogenesis": self.config.neurogenesis
                },
                "metadata": {
                    "creation_time": self.creation_time,
                    "last_update": self.last_update_time,
                    "update_count": self.update_count
                }
            }
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
                
            return True
        except Exception as e:
            print(f"Error saving network: {e}")
            return False
    
    @classmethod
    def load(cls, filepath):
        """
        Load a network from a file.
        
        Creates a new Network instance from a serialized network file.
        
        Args:
            filepath (str): File path to load from
            
        Returns:
            Network: Loaded network or None if loading failed
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            # Create config
            config = Config()
            config.hebbian = data.get("config", {}).get("hebbian", config.hebbian)
            config.neurogenesis = data.get("config", {}).get("neurogenesis", config.neurogenesis)
            
            # Create network
            network = cls(config)
            
            # Restore metadata
            network.creation_time = data.get("metadata", {}).get("creation_time", time.time())
            network.last_update_time = data.get("metadata", {}).get("last_update", time.time())
            network.update_count = data.get("metadata", {}).get("update_count", 0)
            
            # Add neurons
            for name, neuron_data in data.get("neurons", {}).items():
                network.add_neuron(
                    name, 
                    0, 
                    neuron_data.get("position"), 
                    neuron_data.get("type", "default"),
                    neuron_data.get("attributes", {})
                )
            
            # Add connections
            for key, conn_data in data.get("connections", {}).items():
                src, tgt = key.split("_")
                network.connect(src, tgt, conn_data.get("weight", 0))
                
                # Restore creation time
                if (src, tgt) in network.connections:
                    network.connections[(src, tgt)].creation_time = conn_data.get("creation_time", time.time())
            
            # Restore state
            network.state = data.get("state", {})
            
            return network
        except Exception as e:
            print(f"Error loading network: {e}")
            return None