"""
Example demonstrating the visualization of a neural network.
"""

import sys
import os
import time
import random

# Add the parent directory to the path so we can import the library
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PyQt5 import QtWidgets, QtCore
from NeuralNetwork.core import Network, Config
from NeuralNetwork.visualization import NetworkVisualization

class NetworkVisApp(QtWidgets.QMainWindow):
    """Application window for neural network visualization."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neural Network Visualization")
        self.resize(1000, 700)
        
        # Create central widget and layout
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        
        # Create network
        self.network = self.create_network()
        
        # Create visualization widget
        self.vis = NetworkVisualization(self.network)
        layout.addWidget(self.vis)
        
        # Create controls
        controls_layout = QtWidgets.QHBoxLayout()
        
        # Visualization controls
        self.show_weights_cb = QtWidgets.QCheckBox("Show Weights")
        self.show_weights_cb.setChecked(True)
        self.show_weights_cb.stateChanged.connect(self.toggle_weights)
        controls_layout.addWidget(self.show_weights_cb)
        
        self.show_links_cb = QtWidgets.QCheckBox("Show Links")
        self.show_links_cb.setChecked(True)
        self.show_links_cb.stateChanged.connect(self.toggle_links)
        controls_layout.addWidget(self.show_links_cb)
        
        # Network controls
        controls_layout.addStretch()
        
        self.learn_btn = QtWidgets.QPushButton("Perform Learning")
        self.learn_btn.clicked.connect(self.perform_learning)
        controls_layout.addWidget(self.learn_btn)
        
        self.stimulate_btn = QtWidgets.QPushButton("Stimulate Network")
        self.stimulate_btn.clicked.connect(self.stimulate_network)
        controls_layout.addWidget(self.stimulate_btn)
        
        self.add_neuron_btn = QtWidgets.QPushButton("Add Neuron")
        self.add_neuron_btn.clicked.connect(self.add_neuron)
        controls_layout.addWidget(self.add_neuron_btn)
        
        layout.addLayout(controls_layout)
        
        # Setup timer for periodic updates
        self.update_timer = QtCore.QTimer()
        self.update_timer.timeout.connect(self.update_network)
        self.update_timer.start(1000)  # Update every 1 second
        
        # Connect neuron clicked signal
        self.vis.neuronClicked.connect(self.on_neuron_clicked)
    
    def create_network(self):
        """Create and initialize a sample network."""
        # Create config
        config = Config()
        config.hebbian['base_learning_rate'] = 0.2  # Higher learning rate for demo
        config.neurogenesis['cooldown'] = 10  # Shorter cooldown for demo
        
        # Create network
        network = Network(config)
        
        # Add neurons
        network.add_neuron("hunger", 50, (100, 100))
        network.add_neuron("happiness", 50, (300, 100))
        network.add_neuron("cleanliness", 50, (500, 100))
        network.add_neuron("satisfaction", 50, (700, 100))
        network.add_neuron("anxiety", 50, (200, 300))
        network.add_neuron("curiosity", 50, (400, 300))
        network.add_neuron("sleepiness", 50, (600, 300))
        
        # Create connections
        connections = [
            ("hunger", "happiness", -0.3),
            ("hunger", "satisfaction", -0.2),
            ("cleanliness", "happiness", 0.3),
            ("cleanliness", "anxiety", -0.2),
            ("satisfaction", "happiness", 0.5),
            ("anxiety", "happiness", -0.4),
            ("curiosity", "happiness", 0.2),
            ("sleepiness", "curiosity", -0.3)
        ]
        
        for src, dst, weight in connections:
            network.connect(src, dst, weight)
        
        return network
    
    def toggle_weights(self, state):
        """Toggle weight display."""
        self.vis.toggle_weights(state == QtCore.Qt.Checked)
    
    def toggle_links(self, state):
        """Toggle link display."""
        self.vis.toggle_links(state == QtCore.Qt.Checked)
    
    def perform_learning(self):
        """Perform manual learning cycle."""
        updated = self.network.perform_learning()
        if updated:
            QtWidgets.QMessageBox.information(
                self, 
                "Learning Performed",
                f"Updated {len(updated)} neuron pairs"
            )
        else:
            QtWidgets.QMessageBox.information(
                self, 
                "Learning Performed",
                "No neurons were active enough for learning"
            )
    
    def stimulate_network(self):
        """Open dialog to stimulate the network."""
        dialog = StimulateDialog(self.network, self)
        if dialog.exec_():
            self.vis.update()
    
    def add_neuron(self):
        """Add a new neuron to the network."""
        dialog = AddNeuronDialog(self.network, self)
        if dialog.exec_():
            self.vis.update()
    
    def update_network(self):
        """Perform periodic network updates."""
        # Randomly adjust some neuron values
        neuron = random.choice(list(self.network.neurons.keys()))
        value = random.uniform(0, 100)
        self.network.update_state({neuron: value})
        
        # Highlight the changed neuron
        self.vis.highlight_neuron(neuron, 0.5)
        
        # Occasional neurogenesis check
        if random.random() < 0.1:  # 10% chance
            state = {
                "novelty_exposure": random.uniform(0, 5),
                "sustained_stress": random.uniform(0, 1),
                "recent_rewards": random.uniform(0, 1)
            }
            
            if self.network.check_neurogenesis(state):
                # Highlight the new neuron
                new_neuron = self.network.neurogenesis.neurogenesis_data['new_neurons'][-1]
                self.vis.highlight_new_neuron(new_neuron)
    
    def on_neuron_clicked(self, neuron_name):
        """Handle neuron click event."""
        QtWidgets.QMessageBox.information(
            self,
            "Neuron Info",
            f"Neuron: {neuron_name}\n"
            f"Value: {self.network.state.get(neuron_name, 0):.1f}\n"
            f"Type: {self.network.neurons[neuron_name].type}\n"
            f"Connections: {sum(1 for k in self.network.connections.keys() if neuron_name in k)}"
        )


class StimulateDialog(QtWidgets.QDialog):
    """Dialog for stimulating the network."""
    
    def __init__(self, network, parent=None):
        super().__init__(parent)
        self.network = network
        self.setWindowTitle("Stimulate Network")
        self.resize(300, 400)
        
        layout = QtWidgets.QVBoxLayout(self)
        
        # Neuron sliders
        self.sliders = {}
        
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_widget = QtWidgets.QWidget()
        scroll_layout = QtWidgets.QVBoxLayout(scroll_widget)
        
        for name in sorted(network.neurons.keys()):
            group = QtWidgets.QGroupBox(name)
            group_layout = QtWidgets.QVBoxLayout(group)
            
            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            slider.setRange(0, 100)
            slider.setValue(int(network.state.get(name, 0)))
            slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
            slider.setTickInterval(10)
            self.sliders[name] = slider
            
            value_label = QtWidgets.QLabel(f"Value: {slider.value()}")
            slider.valueChanged.connect(lambda v, lbl=value_label: lbl.setText(f"Value: {v}"))
            
            group_layout.addWidget(slider)
            group_layout.addWidget(value_label)
            
            scroll_layout.addWidget(group)
        
        scroll_area.setWidget(scroll_widget)
        layout.addWidget(scroll_area)
        
        # Neurogenesis triggers
        group = QtWidgets.QGroupBox("Neurogenesis Triggers")
        group_layout = QtWidgets.QFormLayout(group)
        
        self.novelty_spin = QtWidgets.QDoubleSpinBox()
        self.novelty_spin.setRange(0, 10)
        self.novelty_spin.setSingleStep(0.1)
        group_layout.addRow("Novelty:", self.novelty_spin)
        
        self.stress_spin = QtWidgets.QDoubleSpinBox()
        self.stress_spin.setRange(0, 2)
        self.stress_spin.setSingleStep(0.1)
        group_layout.addRow("Stress:", self.stress_spin)
        
        self.reward_spin = QtWidgets.QDoubleSpinBox()
        self.reward_spin.setRange(0, 2)
        self.reward_spin.setSingleStep(0.1)
        group_layout.addRow("Reward:", self.reward_spin)
        
        layout.addWidget(group)
        
        # Buttons
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def accept(self):
        """Apply changes when OK is clicked."""
        # Collect values
        values = {}
        
        for name, slider in self.sliders.items():
            values[name] = slider.value()
        
        # Add neurogenesis triggers
        if self.novelty_spin.value() > 0:
            values["novelty_exposure"] = self.novelty_spin.value()
        if self.stress_spin.value() > 0:
            values["sustained_stress"] = self.stress_spin.value()
        if self.reward_spin.value() > 0:
            values["recent_rewards"] = self.reward_spin.value()
        
        # Update network
        self.network.update_state(values)
        
        super().accept()


class AddNeuronDialog(QtWidgets.QDialog):
    """Dialog for adding a new neuron."""
    
    def __init__(self, network, parent=None):
        super().__init__(parent)
        self.network = network
        self.setWindowTitle("Add Neuron")
        self.resize(300, 200)
        
        layout = QtWidgets.QFormLayout(self)
        
        self.name_edit = QtWidgets.QLineEdit()
        layout.addRow("Name:", self.name_edit)
        
        self.type_combo = QtWidgets.QComboBox()
        self.type_combo.addItems(["default", "novelty", "stress", "reward"])
        layout.addRow("Type:", self.type_combo)
        
        self.x_spin = QtWidgets.QSpinBox()
        self.x_spin.setRange(50, 950)
        self.x_spin.setValue(400)
        layout.addRow("X Position:", self.x_spin)
        
        self.y_spin = QtWidgets.QSpinBox()
        self.y_spin.setRange(50, 550)
        self.y_spin.setValue(300)
        layout.addRow("Y Position:", self.y_spin)
        
        self.initial_spin = QtWidgets.QSpinBox()
        self.initial_spin.setRange(0, 100)
        self.initial_spin.setValue(50)
        layout.addRow("Initial Value:", self.initial_spin)
        
        # Buttons
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def validate_and_accept(self):
        """Validate input and add neuron if valid."""
        name = self.name_edit.text().strip()
        
        if not name:
            QtWidgets.QMessageBox.warning(self, "Error", "Name cannot be empty")
            return
        
        if name in self.network.neurons:
            QtWidgets.QMessageBox.warning(self, "Error", f"Neuron '{name}' already exists")
            return
        
        # Add the neuron
        self.network.add_neuron(
            name,
            initial_state=self.initial_spin.value(),
            position=(self.x_spin.value(), self.y_spin.value()),
            neuron_type=self.type_combo.currentText()
        )
        
        # Create initial connections to existing neurons
        for existing in list(self.network.neurons.keys()):
            if existing != name:
                self.network.connect(name, existing, random.uniform(-0.2, 0.2))
        
        self.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = NetworkVisApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()