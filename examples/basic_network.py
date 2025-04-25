"""
Basic neural network example.

This example demonstrates creating a simple network with 
Hebbian learning and neurogenesis.
"""

from neural_network_lib.core import Network, Config
import time
import random


def main():
    # Create a network with default configuration
    config = Config()
    network = Network(config)
    
    print("Creating a simple network with Hebbian learning...")
    
    # Add neurons
    network.add_neuron("input1", initial_state=0, position=(100, 100))
    network.add_neuron("input2", initial_state=0, position=(100, 200))
    network.add_neuron("hidden1", initial_state=0, position=(200, 150))
    network.add_neuron("output", initial_state=0, position=(300, 150))
    
    # Connect neurons
    network.connect("input1", "hidden1", weight=0.2)
    network.connect("input2", "hidden1", weight=0.3)
    network.connect("hidden1", "output", weight=0.5)
    
    # Initialize learning
    network.initialize_learning()
    
    print("Network structure created.")
    print(f"Neurons: {len(network.neurons)}")
    print(f"Connections: {len(network.connections)}")
    
    # Simulation loop
    print("\nRunning simulation...")
    for i in range(5):
        # Stimulate input neurons
        input1_val = random.randint(0, 100)
        input2_val = random.randint(0, 100)
        
        network.update_state({
            "input1": input1_val,
            "input2": input2_val
        })
        
        print(f"\nIteration {i+1}:")
        print(f"  Input1: {input1_val}, Input2: {input2_val}")
        
        # Propagate activation
        network.propagate_activation()
        
        # Get values
        hidden_val = network.get_neuron_value("hidden1")
        output_val = network.get_neuron_value("output")
        
        print(f"  Hidden: {hidden_val:.2f}")
        print(f"  Output: {output_val:.2f}")
        
        # Perform Hebbian learning
        updated_pairs = network.perform_learning()
        if updated_pairs:
            print(f"  Learning: Updated {len(updated_pairs)} connection(s)")
        else:
            print("  Learning: No connections updated")
        
        # Try neurogenesis occasionally
        if i % 2 == 0:
            state = {
                "novelty_exposure": random.uniform(2.0, 4.0),
                "sustained_stress": random.uniform(0.0, 0.5),
                "recent_rewards": random.uniform(0.0, 0.8)
            }
            
            if network.check_neurogenesis(state):
                new_neurons = network.neurogenesis.neurogenesis_data['new_neurons']
                print(f"  Neurogenesis: Created {new_neurons[-1]}")
            else:
                print("  Neurogenesis: No new neurons created")
        
        time.sleep(1)
    
    # Display final network state
    print("\nFinal connection weights:")
    for (src, tgt), conn in network.connections.items():
        print(f"  {src} → {tgt}: {conn.get_weight():.4f}")


if __name__ == "__main__":
    main()