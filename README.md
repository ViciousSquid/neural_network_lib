Neural Network Library
======================

A comprehensive neural network library featuring:

*   Core neural network components (neurons, connections)
*   Multiple learning algorithms:
    *   Hebbian learning ("neurons that fire together, wire together")
    *   Neurogenesis (creation of new neurons during learning)
    *   Backpropagation (supervised learning)
*   Visualization tools with PyQt5

Installation
------------

    # Basic installation
    pip install neural_network_lib
    
    # With visualization support
    pip install neural_network_lib[visualization]
    
    # With examples support (includes matplotlib)
    pip install neural_network_lib[examples]
    
    # With all features
    pip install neural_network_lib[visualization,examples]
    

Features
--------

### Flexible Neural Network Architecture

The library supports various network architectures and learning paradigms:

*   **Basic neural networks** with customizable neuron types and connection weights
*   **Hebbian learning** for unsupervised, biologically-inspired learning
*   **Neurogenesis** for adding new neurons in response to novelty, stress, or reward
*   **Backpropagation** for supervised learning tasks

### Visualization and Interaction

The library includes a complete visualization system built with PyQt5:

*   Interactive visualization of networks
*   GUI for building and experimenting with networks
*   Real-time visualization of learning and activation
*   Layer-based organization of neurons

Quick Start
-----------

    from neural_network_lib.core import Network, Config
    
    # Create a network
    config = Config()
    network = Network(config)
    
    # Add neurons
    network.add_neuron("input1", initial_state=0)
    network.add_neuron("input2", initial_state=0)
    network.add_neuron("output", initial_state=0)
    
    # Connect neurons
    network.connect("input1", "output", weight=0.5)
    network.connect("input2", "output", weight=0.5)
    
    # Update states
    network.update_state({"input1": 100, "input2": 100})
    
    # Propagate activation
    network.propagate_activation()
    
    # Get output
    output_value = network.get_neuron_value("output")
    print(f"Output: {output_value}")
    
    # Initialize learning mechanisms
    network.initialize_learning()
    
    # Perform Hebbian learning
    network.perform_learning()
    

Core Components
---------------

### Network

The central component that manages neurons and connections:

    # Create network
    network = Network()
    
    # Save/load
    network.save("my_network.json")
    loaded_network = Network.load("my_network.json")
    

### Neurons

Basic computational units with various types:

    # Add different types of neurons
    network.add_neuron("input", neuron_type="default")
    network.add_neuron("novelty", neuron_type="novelty")
    network.add_neuron("stress", neuron_type="stress")
    

### Connections

Weighted links between neurons:

    # Create connections with weights
    network.connect("input1", "output", weight=0.5)
    network.connect("input2", "output", weight=-0.3)  # Inhibitory connection
    

Learning Mechanisms
-------------------

### Hebbian Learning

    # Initialize learning
    network.initialize_learning()
    
    # Update states
    network.update_state({"input1": 100, "input2": 100})
    
    # Perform learning
    updated_pairs = network.perform_learning()
    

### Neurogenesis

    # Trigger neurogenesis
    state = {
        "novelty_exposure": 3.5,
        "sustained_stress": 0.2,
        "recent_rewards": 0.8
    }
    network.check_neurogenesis(state)
    

### Backpropagation

    from neural_network_lib.learning import BackpropNetwork
    
    # Create backprop network
    backprop = BackpropNetwork(network)
    
    # Define layers
    backprop.set_layers([
        ["input1", "input2"],  # Input layer
        ["hidden1", "hidden2"],  # Hidden layer
        ["output"]  # Output layer
    ])
    
    # Training data: (inputs, targets)
    training_data = [
        ([0, 0], [0]),
        ([0, 100], [100]),
        ([100, 0], [100]),
        ([100, 100], [0])
    ]
    
    # Train
    errors = backprop.train(training_data, epochs=1000)
    

Visualization
-------------

Using the visualization components requires PyQt5:

    from neural_network_lib.visualization import NetworkVisualization
    from PyQt5 import QtWidgets
    import sys
    
    app = QtWidgets.QApplication(sys.argv)
    
    # Create visualization
    vis = NetworkVisualization(network)
    
    # Display in window
    window = QtWidgets.QMainWindow()
    window.setCentralWidget(vis)
    window.show()
    
    sys.exit(app.exec_())
    

Configuration
-------------

Customize network behavior with configuration parameters:

    config = Config()
    
    # Hebbian learning parameters
    config.hebbian['base_learning_rate'] = 0.2
    config.hebbian['active_threshold'] = 60
    
    # Neurogenesis parameters
    config.neurogenesis['novelty_threshold'] = 2.5
    

Examples
--------

See the `examples/` directory for more detailed examples:

*   `basic_network.py`: Simple network with Hebbian learning
*   `backprop_xor.py`: Solving XOR with backpropagation
*   `visualization_demo.py`: Interactive visualization example

License
-------

MIT