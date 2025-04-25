"""
Neurogenesis module for neural networks.

This module implements neurogenesis, which is the creation of new neurons
in response to various triggers like novelty, stress, or reward.
"""

import time
import random
import math

class Neurogenesis:
    """
    Implements neurogenesis (creation of new neurons) in the network.
    
    Neurogenesis allows the network to grow and adapt to new inputs or
    challenging environments. New neurons are created in response to 
    novelty, stress, or reward signals, with appropriate connections
    to existing neurons.
    
    Attributes:
        network: Reference to the neural network
        config: Configuration parameters
        neurogenesis_data: Tracking data for neurogenesis events
    """
    
    def __init__(self, network):
        """
        Initialize neurogenesis module.
        
        Args:
            network: Neural network to apply neurogenesis to
        """
        self.network = network
        self.config = network.config
        self.neurogenesis_data = {
            'novelty_counter': 0,
            'stress_counter': 0,
            'reward_counter': 0,
            'new_neurons': [],
            'last_neuron_time': time.time()
        }
        
    def check_neurogenesis(self, state):
        """
        Check if conditions for neurogenesis are met and create new neurons if so.
        
        Examines the input state for neurogenesis triggers and creates
        appropriate new neurons when thresholds are exceeded.
        
        Args:
            state (dict): Input state with potential neurogenesis triggers
                          (e.g., 'novelty_exposure', 'sustained_stress', 'recent_rewards')
            
        Returns:
            bool: True if neurogenesis occurred, False otherwise
        """
        current_time = time.time()
        trigger_types = []
        
        # First, apply decay to counters
        decay = self.config.neurogenesis.get('decay_rate', 0.95)
        self.neurogenesis_data['novelty_counter'] *= decay
        self.neurogenesis_data['stress_counter'] *= decay
        self.neurogenesis_data['reward_counter'] *= decay
        
        # Add new state values to counters
        self.neurogenesis_data['novelty_counter'] += state.get('novelty_exposure', 0)
        self.neurogenesis_data['stress_counter'] += state.get('sustained_stress', 0)
        self.neurogenesis_data['reward_counter'] += state.get('recent_rewards', 0)
        
        # Check cooldown
        cooldown = self.config.neurogenesis.get('cooldown', 300)
        if current_time - self.neurogenesis_data['last_neuron_time'] <= cooldown:
            return False
            
        # Check thresholds
        novelty_threshold = self.config.neurogenesis.get('novelty_threshold', 3.0)
        stress_threshold = self.config.neurogenesis.get('stress_threshold', 0.7)
        reward_threshold = self.config.neurogenesis.get('reward_threshold', 0.6)
        
        # Check each trigger type
        if self.neurogenesis_data['novelty_counter'] > novelty_threshold:
            trigger_types.append('novelty')
            self.neurogenesis_data['novelty_counter'] = 0
            
        if self.neurogenesis_data['stress_counter'] > stress_threshold:
            trigger_types.append('stress')
            self.neurogenesis_data['stress_counter'] = 0
            
        if self.neurogenesis_data['reward_counter'] > reward_threshold:
            trigger_types.append('reward')
            self.neurogenesis_data['reward_counter'] = 0
        
        # Create new neurons for each trigger type
        created = False
        for trigger_type in trigger_types:
            new_neuron_name = self._create_neuron(trigger_type, state)
            if new_neuron_name:
                created = True
                
        if created:
            self.neurogenesis_data['last_neuron_time'] = current_time
            
            # Boost learning rate in Hebbian learning if it exists
            if hasattr(self.network, 'learning') and self.network.learning:
                boost = self.config.combined.get('neurogenesis_learning_boost', 1.5)
                self.network.learning.modify_learning_rate(boost)
            
        return created
    
    def _create_neuron(self, neuron_type, state):
        """
        Create a new neuron based on the trigger type.
        
        Args:
            neuron_type (str): Type of trigger ('novelty', 'stress', 'reward')
            state (dict): Current network state
            
        Returns:
            str: Name of the new neuron or None if creation failed
        """
        # Generate neuron name
        base_name = {
            'novelty': 'novel',
            'stress': 'stress',
            'reward': 'reward'
        }.get(neuron_type, 'new')
        
        new_name = f"{base_name}_{len(self.neurogenesis_data['new_neurons'])}"
        
        # Find a good position near active neurons
        position = self._find_position_for_new_neuron(neuron_type, state)
        
        # Add the neuron to the network
        self.network.add_neuron(
            new_name, 
            initial_state=50,  # Start with moderate activation
            position=position, 
            neuron_type=neuron_type
        )
        
        # Create connections to existing neurons
        self._create_connections_for_new_neuron(new_name, neuron_type)
        
        # Add to tracking list
        self.neurogenesis_data['new_neurons'].append(new_name)
        
        return new_name
    
    def _find_position_for_new_neuron(self, neuron_type, state):
        """
        Find a suitable position for a new neuron.
        
        Positions the neuron based on active neurons in the network
        and the neuron type, creating a visually meaningful layout.
        
        Args:
            neuron_type (str): Type of trigger
            state (dict): Current network state
            
        Returns:
            tuple: (x, y) position
        """
        # Find the most active neurons
        active_neurons = sorted(
            [(name, self.network.get_neuron_value(name)) 
             for name in self.network.neurons],
            key=lambda x: x[1],
            reverse=True
        )[:3]  # Top 3 most active
        
        if active_neurons:
            # Use the most active neuron as anchor
            anchor_name = active_neurons[0][0]
            anchor = self.network.neurons[anchor_name]
            base_x, base_y = anchor.get_position()
            
            # Add a random offset in the direction suggested by neuron type
            angle = random.uniform(0, 2 * math.pi)
            distance = random.uniform(50, 100)
            
            # Adjust angle based on neuron type for visual clustering
            if neuron_type == 'novelty':
                angle += math.pi / 6  # 30 degrees bias
            elif neuron_type == 'stress':
                angle += math.pi / 2  # 90 degrees bias
            elif neuron_type == 'reward':
                angle += math.pi  # 180 degrees bias
                
            x = base_x + math.cos(angle) * distance
            y = base_y + math.sin(angle) * distance
            
            return (x, y)
        else:
            # Fall back to random position if no active neurons
            return (
                random.uniform(100, 900),
                random.uniform(100, 500)
            )
    
    def _create_connections_for_new_neuron(self, new_name, neuron_type):
        """
        Create connections between the new neuron and existing ones.
        
        Creates appropriate connections with weights based on neuron type,
        with stronger connections to related concept neurons.
        
        Args:
            new_name (str): Name of the new neuron
            neuron_type (str): Type of the new neuron
        """
        conn_strength = self.config.neurogenesis.get('new_neuron_connection_strength', 0.3)
        
        # Define related neurons that should have stronger connections
        related_neurons = {
            'novelty': ['curiosity'],
            'stress': ['anxiety'],
            'reward': ['satisfaction', 'happiness']
        }.get(neuron_type, [])
        
        # Connect to all existing neurons
        for existing_name in self.network.neurons:
            if existing_name != new_name and existing_name not in self.network.excluded_neurons:
                # Determine base weight
                if existing_name in related_neurons:
                    # Stronger connection to related neurons
                    weight = conn_strength * 2.0
                else:
                    # Random weight for other neurons
                    weight = random.uniform(-conn_strength, conn_strength)
                
                # Create bidirectional connection
                self.network.connect(new_name, existing_name, weight, bidirectional=True)
    
    def get_neurogenesis_stats(self):
        """
        Get statistics about neurogenesis.
        
        Returns:
            dict: Neurogenesis statistics including counters and new neurons
        """
        return {
            'total_new_neurons': len(self.neurogenesis_data['new_neurons']),
            'last_created': time.time() - self.neurogenesis_data['last_neuron_time'],
            'novelty_counter': self.neurogenesis_data['novelty_counter'],
            'stress_counter': self.neurogenesis_data['stress_counter'],
            'reward_counter': self.neurogenesis_data['reward_counter'],
            'new_neuron_names': self.neurogenesis_data['new_neurons']
        }
    
    def reset_counters(self):
        """
        Reset neurogenesis trigger counters.
        
        Useful for starting fresh after significant changes to the environment.
        """
        self.neurogenesis_data['novelty_counter'] = 0
        self.neurogenesis_data['stress_counter'] = 0
        self.neurogenesis_data['reward_counter'] = 0