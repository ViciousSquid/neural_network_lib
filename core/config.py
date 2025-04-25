"""
Configuration module for neural networks.

This module defines the Config class, which contains parameters
for various aspects of neural network behavior.
"""

import json
import os

class Config:
    """
    Configuration parameters for the neural network.
    
    Contains settings for Hebbian learning, neurogenesis, and other
    network properties. Parameters can be loaded from and saved to
    JSON files for easy reuse.
    
    Attributes:
        hebbian (dict): Hebbian learning parameters
        neurogenesis (dict): Neurogenesis parameters
        combined (dict): Additional combined parameters
    """
    
    def __init__(self):
        """
        Initialize with default configuration values.
        
        Sets up reasonable defaults for all parameters, which can
        later be overridden by loading from a file or direct modification.
        """
        # Hebbian learning parameters
        self.hebbian = {
            'base_learning_rate': 0.1,        # Base rate for Hebbian learning
            'threshold': 0.7,                 # Activation threshold for learning
            'weight_decay': 0.01,             # Weight decay factor (prevents runaway weights)
            'max_weight': 1.0,                # Maximum connection weight
            'min_weight': -1.0,               # Minimum connection weight
            'learning_interval': 30000,       # Milliseconds between learning cycles
            'active_threshold': 50            # Threshold to consider a neuron active (0-100)
        }
        
        # Neurogenesis parameters
        self.neurogenesis = {
            'novelty_threshold': 3.0,         # Threshold for novelty-triggered neurogenesis
            'stress_threshold': 0.7,          # Threshold for stress-triggered neurogenesis
            'reward_threshold': 0.6,          # Threshold for reward-triggered neurogenesis
            'cooldown': 300,                  # Seconds between neurogenesis events
            'decay_rate': 0.95,               # Decay rate for trigger counters
            'new_neuron_connection_strength': 0.3,  # Initial connection strength for new neurons
            'highlight_duration': 5.0         # Duration to highlight new neurons (seconds)
        }
        
        # Additional combined parameters for advanced functionality
        self.combined = {
            'neurogenesis_learning_boost': 1.5,  # Learning rate boost after neurogenesis
            'goal_reinforcement_factor': 2.0     # Multiplier for goal-oriented learning
        }
    
    def load_from_file(self, file_path):
        """
        Load configuration from a JSON file.
        
        Args:
            file_path (str): Path to the config file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with open(file_path, 'r') as f:
                config = json.load(f)
                
                if 'hebbian' in config:
                    self.hebbian.update(config['hebbian'])
                if 'neurogenesis' in config:
                    self.neurogenesis.update(config['neurogenesis'])
                if 'combined' in config:
                    self.combined.update(config['combined'])
                    
            return True
        except FileNotFoundError:
            print(f"Config file {file_path} not found, using defaults")
            return False
        except json.JSONDecodeError:
            print(f"Invalid config file {file_path}, using defaults")
            return False
        except Exception as e:
            print(f"Error loading config: {e}")
            return False
            
    def save_to_file(self, file_path):
        """
        Save configuration to a JSON file.
        
        Args:
            file_path (str): Path to save the config file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            config = {
                'hebbian': self.hebbian,
                'neurogenesis': self.neurogenesis,
                'combined': self.combined
            }
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            with open(file_path, 'w') as f:
                json.dump(config, f, indent=2)
                
            return True
        except Exception as e:
            print(f"Error saving config: {e}")
            return False