# network_builder_gui.py

import sys
import os
import random
import time
from PyQt5 import QtWidgets, QtCore, QtGui

# Ensure library is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from NeuralNetwork.core import Network, Config, Neuron, Connection
from NeuralNetwork.visualization import NetworkVisualization

class NetworkBuilderGUI(QtWidgets.QMainWindow):
    """
    GUI application for building and visualizing neural networks.
    """
    
    # Define the event filter as a nested class
    class VisualizationEventFilter(QtCore.QObject):
        def __init__(self, parent):
            super().__init__(parent)
            self.parent = parent
        
        def eventFilter(self, watched, event):
            if event.type() == QtCore.QEvent.MouseButtonPress:
                self.parent.visualization_mouse_press(event)
                return False  # Allow event to propagate to original handler
            elif event.type() == QtCore.QEvent.MouseMove:
                self.parent.visualization_mouse_move(event)
                return False
            elif event.type() == QtCore.QEvent.MouseButtonRelease:
                self.parent.visualization_mouse_release(event)
                return False
            return False
    
    def __init__(self):
        super().__init__()
        self.network = Network()
        self.setup_ui()
        self.mode = "select"  # Current editor mode: select, add_neuron, add_connection, etc.
        self.selected_neuron = None
        self.connection_start = None
        self.neuron_counter = 0
        self.layer_counter = 0

    def select_color_dialog(self, button, color_var):
        """Open color picker dialog and update the button color."""
        color = QtWidgets.QColorDialog.getColor(color_var, self, "Select Layer Color")
        
        if color.isValid():
            # Update the reference variable
            color_var.setRgb(color.red(), color.green(), color.blue())
            
            # Set button background color
            button.setStyleSheet(f"background-color: {color.name()}")
    
    def setup_ui(self):
        """Set up the main UI components."""
        self.setWindowTitle("Neural Network Builder")
        self.setGeometry(100, 100, 1200, 800)
        
        # Central widget with splitter
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        
        main_layout = QtWidgets.QHBoxLayout(self.central_widget)
        
        # Splitter for resizable panels
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_layout.addWidget(self.splitter)
        
        # Left panel for controls
        self.control_panel = QtWidgets.QWidget()
        control_layout = QtWidgets.QVBoxLayout(self.control_panel)
        
        # Create tools group
        self.create_tools_group(control_layout)
        
        # Layer tools group
        self.create_layer_tools_group(control_layout)
        
        # Properties panel
        self.create_properties_panel(control_layout)
        
        # Simulation controls
        self.create_simulation_controls(control_layout)
        
        # Add a spacer to push everything up
        control_layout.addStretch()
        
        # Network visualization (right panel)
        self.vis_panel = QtWidgets.QWidget()
        vis_layout = QtWidgets.QVBoxLayout(self.vis_panel)
        
        # Create network visualization
        self.vis = NetworkVisualization(self.network)
        self.vis.setMinimumSize(800, 600)
        self.vis.neuronClicked.connect(self.on_neuron_clicked)
        self.vis.setMouseTracking(True)
        
        # Install event filter instead of directly overriding mouse events
        self.vis_event_filter = self.VisualizationEventFilter(self)
        self.vis.installEventFilter(self.vis_event_filter)
        
        vis_layout.addWidget(self.vis)
        
        # Add panels to splitter
        self.splitter.addWidget(self.control_panel)
        self.splitter.addWidget(self.vis_panel)
        self.splitter.setSizes([300, 900])  # Initial sizes
        
        # Create menu bar
        self.create_menu_bar()

        # Add zoom slider at the bottom of the window
        zoom_layout = QtWidgets.QHBoxLayout()
        zoom_label = QtWidgets.QLabel("Zoom:")
        self.zoom_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.zoom_slider.setRange(50, 200)  # 50% to 200% zoom
        self.zoom_slider.setValue(100)  # Default 100%
        self.zoom_slider.setTickInterval(25)
        self.zoom_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.zoom_slider.valueChanged.connect(self.on_zoom_changed)
        
        zoom_layout.addWidget(zoom_label)
        zoom_layout.addWidget(self.zoom_slider)
        
        # Add the zoom controls to the main layout
        vis_layout.addLayout(zoom_layout)
        
        # Status bar
        self.statusBar().showMessage("Ready")

    def on_zoom_changed(self, value):
        """Handle zoom slider value changes."""
        zoom_factor = value / 100.0
        self.vis.set_zoom(zoom_factor)
    
    def create_tools_group(self, layout):
        """Create the network editing tools group."""
        group = QtWidgets.QGroupBox("Tools")
        group_layout = QtWidgets.QVBoxLayout(group)
        
        # Mode selection buttons
        self.select_btn = QtWidgets.QPushButton("Select")
        self.select_btn.setCheckable(True)
        self.select_btn.setChecked(True)
        self.select_btn.clicked.connect(lambda: self.set_mode("select"))
        group_layout.addWidget(self.select_btn)
        
        self.add_neuron_btn = QtWidgets.QPushButton("Add Neuron")
        self.add_neuron_btn.setCheckable(True)
        self.add_neuron_btn.clicked.connect(lambda: self.set_mode("add_neuron"))
        group_layout.addWidget(self.add_neuron_btn)
        
        self.add_connection_btn = QtWidgets.QPushButton("Add Connection")
        self.add_connection_btn.setCheckable(True)
        self.add_connection_btn.clicked.connect(lambda: self.set_mode("add_connection"))
        group_layout.addWidget(self.add_connection_btn)
        
        self.remove_btn = QtWidgets.QPushButton("Remove")
        self.remove_btn.setCheckable(True)
        self.remove_btn.clicked.connect(lambda: self.set_mode("remove"))
        group_layout.addWidget(self.remove_btn)
        
        self.mode_buttons = [
            self.select_btn, 
            self.add_neuron_btn, 
            self.add_connection_btn, 
            self.remove_btn
        ]
        
        # Neuron type selection
        neuron_type_layout = QtWidgets.QHBoxLayout()
        self.neuron_type_label = QtWidgets.QLabel("Neuron Type:")
        self.neuron_type_combo = QtWidgets.QComboBox()
        self.neuron_type_combo.addItems(["default", "novelty", "stress", "reward"])
        neuron_type_layout.addWidget(self.neuron_type_label)
        neuron_type_layout.addWidget(self.neuron_type_combo)
        group_layout.addLayout(neuron_type_layout)
        
        layout.addWidget(group)
    
    def create_layer_tools_group(self, layout):
        """Create the layer creation tools group."""
        group = QtWidgets.QGroupBox("Layer Tools")
        group_layout = QtWidgets.QVBoxLayout(group)
        
        # Add layer button with parameters
        self.add_layer_btn = QtWidgets.QPushButton("Add Layer")
        self.add_layer_btn.clicked.connect(self.add_layer_dialog)
        group_layout.addWidget(self.add_layer_btn)
        
        # Connect layers button
        self.connect_layers_btn = QtWidgets.QPushButton("Connect Layers")
        self.connect_layers_btn.clicked.connect(self.connect_layers_dialog)
        group_layout.addWidget(self.connect_layers_btn)
        
        # Feedforward network button
        self.create_feedforward_btn = QtWidgets.QPushButton("Create Feedforward Network")
        self.create_feedforward_btn.clicked.connect(self.create_feedforward_dialog)
        group_layout.addWidget(self.create_feedforward_btn)
        
        # Auto-layout button
        self.auto_layout_btn = QtWidgets.QPushButton("Auto-Layout Network")
        self.auto_layout_btn.clicked.connect(self.auto_layout_network)
        group_layout.addWidget(self.auto_layout_btn)
        
        layout.addWidget(group)
    
    def create_properties_panel(self, layout):
        """Create the properties panel for neuron/connection editing."""
        group = QtWidgets.QGroupBox("Properties")
        group.setStyleSheet("QGroupBox { background-color: rgba(240, 240, 240, 0.7); border-radius: 5px; }")
        group_layout = QtWidgets.QVBoxLayout(group)
        
        # Neuron properties
        self.neuron_props_widget = QtWidgets.QWidget()
        neuron_layout = QtWidgets.QFormLayout(self.neuron_props_widget)
        
        self.neuron_name_edit = QtWidgets.QLineEdit()
        self.neuron_name_edit.textChanged.connect(self.update_neuron_property)
        neuron_layout.addRow("Name:", self.neuron_name_edit)
        
        self.neuron_value_spin = QtWidgets.QSpinBox()
        self.neuron_value_spin.setRange(0, 100)
        self.neuron_value_spin.valueChanged.connect(self.update_neuron_property)
        neuron_layout.addRow("Value:", self.neuron_value_spin)
        
        self.neuron_type_edit = QtWidgets.QComboBox()
        self.neuron_type_edit.addItems(["default", "novelty", "stress", "reward"])
        self.neuron_type_edit.currentTextChanged.connect(self.update_neuron_property)
        neuron_layout.addRow("Type:", self.neuron_type_edit)
        
        self.neuron_x_spin = QtWidgets.QSpinBox()
        self.neuron_x_spin.setRange(0, 2000)
        self.neuron_x_spin.valueChanged.connect(self.update_neuron_property)
        neuron_layout.addRow("X Position:", self.neuron_x_spin)
        
        self.neuron_y_spin = QtWidgets.QSpinBox()
        self.neuron_y_spin.setRange(0, 2000)
        self.neuron_y_spin.valueChanged.connect(self.update_neuron_property)
        neuron_layout.addRow("Y Position:", self.neuron_y_spin)
        
        # Connection properties
        self.connection_props_widget = QtWidgets.QWidget()
        connection_layout = QtWidgets.QFormLayout(self.connection_props_widget)
        
        self.connection_source_label = QtWidgets.QLabel()
        connection_layout.addRow("Source:", self.connection_source_label)
        
        self.connection_target_label = QtWidgets.QLabel()
        connection_layout.addRow("Target:", self.connection_target_label)
        
        self.connection_weight_spin = QtWidgets.QDoubleSpinBox()
        self.connection_weight_spin.setRange(-1.0, 1.0)
        self.connection_weight_spin.setSingleStep(0.05)
        self.connection_weight_spin.valueChanged.connect(self.update_connection_property)
        connection_layout.addRow("Weight:", self.connection_weight_spin)
        
        # Add to stacked widget
        self.properties_stack = QtWidgets.QStackedWidget()
        self.properties_stack.addWidget(self.neuron_props_widget)
        self.properties_stack.addWidget(self.connection_props_widget)
        
        # Add placeholder for when nothing is selected
        nothing_selected = QtWidgets.QLabel("No item selected")
        nothing_selected.setAlignment(QtCore.Qt.AlignCenter)
        self.properties_stack.addWidget(nothing_selected)
        self.properties_stack.setCurrentIndex(2)  # Start with nothing selected
        
        group_layout.addWidget(self.properties_stack)
        layout.addWidget(group)
    
    def create_simulation_controls(self, layout):
        """Create the simulation control panel."""
        group = QtWidgets.QGroupBox("Simulation")
        group_layout = QtWidgets.QVBoxLayout(group)
        
        # Update state
        update_state_layout = QtWidgets.QHBoxLayout()
        self.update_neuron_combo = QtWidgets.QComboBox()
        self.update_neuron_combo.addItem("Select Neuron")
        self.update_value_spin = QtWidgets.QSpinBox()
        self.update_value_spin.setRange(0, 100)
        self.update_state_btn = QtWidgets.QPushButton("Update")
        self.update_state_btn.clicked.connect(self.update_neuron_state)
        
        update_state_layout.addWidget(self.update_neuron_combo)
        update_state_layout.addWidget(self.update_value_spin)
        update_state_layout.addWidget(self.update_state_btn)
        group_layout.addLayout(update_state_layout)
        
        # Learning and propagation
        self.perform_learning_btn = QtWidgets.QPushButton("Perform Learning")
        self.perform_learning_btn.clicked.connect(self.perform_learning)
        group_layout.addWidget(self.perform_learning_btn)
        
        self.propagate_btn = QtWidgets.QPushButton("Propagate Activation")
        self.propagate_btn.clicked.connect(self.propagate_activation)
        group_layout.addWidget(self.propagate_btn)
        
        # Neurogenesis
        self.trigger_neurogenesis_btn = QtWidgets.QPushButton("Trigger Neurogenesis")
        self.trigger_neurogenesis_btn.clicked.connect(self.trigger_neurogenesis)
        group_layout.addWidget(self.trigger_neurogenesis_btn)
        
        # Auto-simulation
        self.auto_sim_check = QtWidgets.QCheckBox("Auto-Simulation")
        self.auto_sim_check.stateChanged.connect(self.toggle_auto_simulation)
        group_layout.addWidget(self.auto_sim_check)
        
        layout.addWidget(group)
        
        # Create timer for auto-simulation
        self.auto_sim_timer = QtCore.QTimer()
        self.auto_sim_timer.timeout.connect(self.auto_simulation_step)
    
    def create_menu_bar(self):
        """Create the application menu bar."""
        menu_bar = self.menuBar()
        
        # File menu
        file_menu = menu_bar.addMenu("File")
        
        new_action = QtWidgets.QAction("New Network", self)
        new_action.triggered.connect(self.new_network)
        file_menu.addAction(new_action)
        
        open_action = QtWidgets.QAction("Open Network", self)
        open_action.triggered.connect(self.open_network)
        file_menu.addAction(open_action)
        
        save_action = QtWidgets.QAction("Save Network", self)
        save_action.triggered.connect(self.save_network)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QtWidgets.QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Edit menu
        edit_menu = menu_bar.addMenu("Edit")
        
        clear_action = QtWidgets.QAction("Clear Network", self)
        clear_action.triggered.connect(self.clear_network)
        edit_menu.addAction(clear_action)
        
        randomize_action = QtWidgets.QAction("Randomize Weights", self)
        randomize_action.triggered.connect(self.randomize_weights)
        edit_menu.addAction(randomize_action)
        
        # View menu
        view_menu = menu_bar.addMenu("View")
        
        toggle_weights_action = QtWidgets.QAction("Show Weights", self)
        toggle_weights_action.setCheckable(True)
        toggle_weights_action.setChecked(True)
        toggle_weights_action.triggered.connect(lambda checked: self.vis.toggle_weights(checked))
        view_menu.addAction(toggle_weights_action)
        
        toggle_links_action = QtWidgets.QAction("Show Links", self)
        toggle_links_action.setCheckable(True)
        toggle_links_action.setChecked(True)
        toggle_links_action.triggered.connect(lambda checked: self.vis.toggle_links(checked))
        view_menu.addAction(toggle_links_action)
        
        # Help menu
        help_menu = menu_bar.addMenu("Help")
        
        about_action = QtWidgets.QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def set_mode(self, mode):
        """Change the current editor mode."""
        self.mode = mode
        
        # Update button states
        for button in self.mode_buttons:
            button.setChecked(False)
        
        if mode == "select":
            self.select_btn.setChecked(True)
            self.statusBar().showMessage("Select Mode: Click to select neurons or connections")
        elif mode == "add_neuron":
            self.add_neuron_btn.setChecked(True)
            self.statusBar().showMessage("Add Neuron Mode: Click to add new neurons")
        elif mode == "add_connection":
            self.add_connection_btn.setChecked(True)
            self.statusBar().showMessage("Add Connection Mode: Click and drag between neurons")
        elif mode == "remove":
            self.remove_btn.setChecked(True)
            self.statusBar().showMessage("Remove Mode: Click to remove neurons or connections")
        
        # Clear selection when changing modes
        self.selected_neuron = None
        self.connection_start = None
        self.properties_stack.setCurrentIndex(2)  # Show 'nothing selected'
        self.vis.update()
    
    # Mouse event handlers for the visualization widget
    def visualization_mouse_press(self, event):
        """Handle mouse press events in the visualization."""
        # Calculate logical coordinates in the visualization
        scale_x = self.vis.width() / 1200
        scale_y = self.vis.height() / 600
        scale = min(scale_x, scale_y)
        
        pos_x = event.x() / scale
        pos_y = event.y() / scale
        
        if event.button() == QtCore.Qt.LeftButton:
            if self.mode == "select":
                # Try to select a neuron
                for name, neuron in self.network.neurons.items():
                    neuron_pos = neuron.get_position()
                    distance = ((pos_x - neuron_pos[0])**2 + (pos_y - neuron_pos[1])**2)**0.5
                    
                    if distance <= 25:  # Neuron radius
                        self.selected_neuron = name
                        self.update_property_panel(name, "neuron")
                        self.vis.highlight_neuron(name, 0.5)
                        self.statusBar().showMessage(f"Selected neuron: {name}")
                        break
                else:
                    # If no neuron was clicked, try to select a connection
                    for (src, tgt), conn in self.network.connections.items():
                        src_pos = self.network.neurons[src].get_position()
                        tgt_pos = self.network.neurons[tgt].get_position()
                        
                        # Check if click is near the connection line
                        if self.is_point_near_line(pos_x, pos_y, src_pos[0], src_pos[1], tgt_pos[0], tgt_pos[1]):
                            self.selected_connection = (src, tgt)
                            self.update_property_panel((src, tgt), "connection")
                            self.statusBar().showMessage(f"Selected connection: {src} → {tgt}")
                            break
                    else:
                        # If no neuron or connection was clicked, clear selection
                        self.selected_neuron = None
                        self.selected_connection = None
                        self.properties_stack.setCurrentIndex(2)  # Show 'nothing selected'
                        self.statusBar().showMessage("No item selected")
            
            elif self.mode == "add_neuron":
                # Add a new neuron at the click position
                self.add_neuron_at_position(pos_x, pos_y)
            
            elif self.mode == "add_connection":
                # Start drawing a connection from a neuron
                for name, neuron in self.network.neurons.items():
                    neuron_pos = neuron.get_position()
                    distance = ((pos_x - neuron_pos[0])**2 + (pos_y - neuron_pos[1])**2)**0.5
                    
                    if distance <= 25:  # Neuron radius
                        self.connection_start = name
                        self.statusBar().showMessage(f"Drawing connection from: {name}")
                        break
            
            elif self.mode == "remove":
                # Try to remove a neuron
                for name, neuron in self.network.neurons.items():
                    neuron_pos = neuron.get_position()
                    distance = ((pos_x - neuron_pos[0])**2 + (pos_y - neuron_pos[1])**2)**0.5
                    
                    if distance <= 25:  # Neuron radius
                        self.remove_neuron(name)
                        break
                else:
                    # If no neuron was clicked, try to remove a connection
                    for (src, tgt), conn in list(self.network.connections.items()):
                        src_pos = self.network.neurons[src].get_position()
                        tgt_pos = self.network.neurons[tgt].get_position()
                        
                        # Check if click is near the connection line
                        if self.is_point_near_line(pos_x, pos_y, src_pos[0], src_pos[1], tgt_pos[0], tgt_pos[1]):
                            self.remove_connection(src, tgt)
                            break
        
        # Handle original visualization mouse press (for dragging neurons)
        if hasattr(QtWidgets.QWidget, "mousePressEvent"):
            QtWidgets.QWidget.mousePressEvent(self.vis, event)
    
    def visualization_mouse_move(self, event):
        """Handle mouse move events in the visualization."""
        # No forwarding - let original handler process the event
        pass
    
    def visualization_mouse_release(self, event):
        """Handle mouse release events in the visualization."""
        if event.button() == QtCore.Qt.LeftButton and self.mode == "add_connection" and self.connection_start:
            # Calculate logical coordinates in the visualization
            scale_x = self.vis.width() / 1200
            scale_y = self.vis.height() / 600
            scale = min(scale_x, scale_y)
            
            pos_x = event.x() / scale
            pos_y = event.y() / scale
            
            # Check if released on a neuron
            for name, neuron in self.network.neurons.items():
                if name == self.connection_start:
                    continue  # Skip the source neuron
                    
                neuron_pos = neuron.get_position()
                distance = ((pos_x - neuron_pos[0])**2 + (pos_y - neuron_pos[1])**2)**0.5
                
                if distance <= 25:  # Neuron radius
                    # Create a connection
                    weight = 0.2  # Default weight
                    self.network.connect(self.connection_start, name, weight)
                    self.statusBar().showMessage(f"Created connection: {self.connection_start} → {name}")
                    break
            
            # Reset connection start
            self.connection_start = None
        
        # Forward to original event handler
        if hasattr(QtWidgets.QWidget, "mouseReleaseEvent"):
            QtWidgets.QWidget.mouseReleaseEvent(self.vis, event)
    
    def on_neuron_clicked(self, neuron_name):
        """Handle neuron click signal from visualization widget."""
        if self.mode == "select":
            self.selected_neuron = neuron_name
            self.update_property_panel(neuron_name, "neuron")
    
    def update_property_panel(self, item, item_type):
        """Update the property panel for the selected item."""
        if item_type == "neuron":
            # Update neuron properties panel
            self.properties_stack.setCurrentIndex(0)  # Show neuron properties
            
            # Update fields without triggering callbacks
            self.neuron_name_edit.blockSignals(True)
            self.neuron_value_spin.blockSignals(True)
            self.neuron_type_edit.blockSignals(True)
            self.neuron_x_spin.blockSignals(True)
            self.neuron_y_spin.blockSignals(True)
            
            neuron = self.network.neurons[item]
            self.neuron_name_edit.setText(item)
            self.neuron_value_spin.setValue(int(self.network.state.get(item, 0)))
            self.neuron_type_edit.setCurrentText(neuron.type)
            
            pos = neuron.get_position()
            self.neuron_x_spin.setValue(int(pos[0]))
            self.neuron_y_spin.setValue(int(pos[1]))
            
            self.neuron_name_edit.blockSignals(False)
            self.neuron_value_spin.blockSignals(False)
            self.neuron_type_edit.blockSignals(False)
            self.neuron_x_spin.blockSignals(False)
            self.neuron_y_spin.blockSignals(False)
        
        elif item_type == "connection":
            # Update connection properties panel
            self.properties_stack.setCurrentIndex(1)  # Show connection properties
            
            src, tgt = item
            self.connection_source_label.setText(src)
            self.connection_target_label.setText(tgt)
            
            # Update weight field without triggering callback
            self.connection_weight_spin.blockSignals(True)
            weight = self.network.connections[item].get_weight()
            self.connection_weight_spin.setValue(weight)
            self.connection_weight_spin.blockSignals(False)
    
    def update_neuron_property(self):
        """Update neuron properties when changed in the panel."""
        if not self.selected_neuron:
            return
        
        # Get current neuron
        neuron = self.network.neurons[self.selected_neuron]
        
        # Update position
        x = self.neuron_x_spin.value()
        y = self.neuron_y_spin.value()
        neuron.set_position(x, y)
        
        # Update type
        new_type = self.neuron_type_edit.currentText()
        neuron.type = new_type
        
        # Update value
        value = self.neuron_value_spin.value()
        self.network.state[self.selected_neuron] = value
        
        # Update name (more complex, as it requires updating references)
        new_name = self.neuron_name_edit.text()
        if new_name != self.selected_neuron and new_name not in self.network.neurons:
            # Create a new neuron with the new name
            new_neuron = Neuron(new_name, neuron.get_position(), neuron.type, neuron.attributes)
            
            # Copy state
            self.network.state[new_name] = self.network.state.get(self.selected_neuron, 0)
            
            # Update connections
            connections_to_update = []
            
            for (src, tgt), conn in list(self.network.connections.items()):
                if src == self.selected_neuron:
                    connections_to_update.append((new_name, tgt, conn.get_weight()))
                    del self.network.connections[(src, tgt)]
                
                if tgt == self.selected_neuron:
                    connections_to_update.append((src, new_name, conn.get_weight()))
                    del self.network.connections[(src, tgt)]
            
            # Add the new connections
            for src, tgt, weight in connections_to_update:
                self.network.connect(src, tgt, weight)
            
            # Update any layers that contain this neuron
            if hasattr(self, 'layers'):
                for layer_name, layer_info in self.layers.items():
                    if isinstance(layer_info, dict) and 'neurons' in layer_info:
                        neurons = layer_info['neurons']
                    else:
                        neurons = layer_info  # Handle old format
                        
                    if self.selected_neuron in neurons:
                        idx = neurons.index(self.selected_neuron)
                        neurons[idx] = new_name
            
            # Update selection if in the selected_neurons set
            if hasattr(self.vis, 'selected_neurons') and self.selected_neuron in self.vis.selected_neurons:
                self.vis.selected_neurons.remove(self.selected_neuron)
                self.vis.selected_neurons.add(new_name)
            
            # Remove old neuron and add new one
            self.network.neurons[new_name] = new_neuron
            del self.network.neurons[self.selected_neuron]
            
            # Remove old state
            if self.selected_neuron in self.network.state:
                del self.network.state[self.selected_neuron]
            
            # Update selected neuron
            self.selected_neuron = new_name
            
            # Update neuron combo box
            index = self.update_neuron_combo.findText(self.selected_neuron)
            if index >= 0:
                self.update_neuron_combo.removeItem(index)
            self.update_neuron_combo.addItem(new_name)
        
        # Update visualization
        self.vis.update_layer_boxes()  # Update layer boxes
        self.vis.update()
    
    def update_connection_property(self):
        """Update connection properties when changed in the panel."""
        if not hasattr(self, 'selected_connection') or not self.selected_connection:
            return
        
        # Get current connection
        src, tgt = self.selected_connection
        if (src, tgt) in self.network.connections:
            conn = self.network.connections[(src, tgt)]
            
            # Update weight
            weight = self.connection_weight_spin.value()
            conn.set_weight(weight)
            
            # Update visualization
            self.vis.update()
    
    def add_neuron_at_position(self, x, y):
        """Add a new neuron at the specified position."""
        # Generate a unique name
        name = f"neuron_{self.neuron_counter}"
        self.neuron_counter += 1
        
        # Get neuron type from selection
        neuron_type = self.neuron_type_combo.currentText()
        
        # Add the neuron
        self.network.add_neuron(name, 50, (x, y), neuron_type)
        
        # Update visualization
        self.vis.highlight_new_neuron(name, 1.0)
        self.vis.update()
        
        # Update neuron combo box
        self.update_neuron_combo.addItem(name)
        
        self.statusBar().showMessage(f"Added neuron: {name}")
    
    def remove_neuron(self, name):
        """Remove a neuron and its connections."""
        # Remove connections involving this neuron
        to_remove = []
        for key in self.network.connections.keys():
            if name in key:
                to_remove.append(key)
        
        for key in to_remove:
            del self.network.connections[key]
        
        # Remove from state
        if name in self.network.state:
            del self.network.state[name]
        
        # Remove from neurons
        del self.network.neurons[name]
        
        # Update visualization
        self.vis.update()
        
        # Update neuron combo box
        index = self.update_neuron_combo.findText(name)
        if index >= 0:
            self.update_neuron_combo.removeItem(index)
        
        # Clear selection if the removed neuron was selected
        if self.selected_neuron == name:
            self.selected_neuron = None
            self.properties_stack.setCurrentIndex(2)  # Show 'nothing selected'
        
        self.statusBar().showMessage(f"Removed neuron: {name}")
    
    def remove_connection(self, src, tgt):
        """Remove a connection."""
        # Remove the connection
        if (src, tgt) in self.network.connections:
            del self.network.connections[(src, tgt)]
        
        # Also check for reverse connection in case it's bidirectional
        if (tgt, src) in self.network.connections:
            del self.network.connections[(tgt, src)]
        
        # Update visualization
        self.vis.update()
        
        # Clear selection if the removed connection was selected
        if hasattr(self, 'selected_connection') and self.selected_connection in [(src, tgt), (tgt, src)]:
            self.selected_connection = None
            self.properties_stack.setCurrentIndex(2)  # Show 'nothing selected'
        
        self.statusBar().showMessage(f"Removed connection: {src} ↔ {tgt}")
    
    def is_point_near_line(self, x, y, x1, y1, x2, y2, threshold=10):
        """Check if a point is near a line segment."""
        # Calculate the distance from point to line segment
        line_length = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
        if line_length == 0:
            return False
        
        # Calculate the normalized dot product
        t = max(0, min(1, ((x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)) / (line_length**2)))
        
        # Calculate the closest point on the line segment
        closest_x = x1 + t * (x2 - x1)
        closest_y = y1 + t * (y2 - y1)
        
        # Calculate the distance from the point to the closest point
        distance = ((x - closest_x)**2 + (y - closest_y)**2)**0.5
        
        return distance <= threshold
    
    def add_layer_dialog(self):
        """Show dialog to add a layer of neurons."""
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Add Layer")
        dialog.setMinimumWidth(300)
        
        layout = QtWidgets.QFormLayout(dialog)
        
        prefix_edit = QtWidgets.QLineEdit(f"layer{self.layer_counter}_")
        layout.addRow("Neuron Prefix:", prefix_edit)
        
        count_spin = QtWidgets.QSpinBox()
        count_spin.setRange(1, 20)
        count_spin.setValue(3)
        layout.addRow("Number of Neurons:", count_spin)
        
        x_pos_spin = QtWidgets.QSpinBox()
        x_pos_spin.setRange(50, 2000)
        x_pos_spin.setValue(100 + self.layer_counter * 200)
        layout.addRow("X Position:", x_pos_spin)
        
        y_start_spin = QtWidgets.QSpinBox()
        y_start_spin.setRange(50, 2000)
        y_start_spin.setValue(100)
        layout.addRow("Y Start Position:", y_start_spin)
        
        spacing_spin = QtWidgets.QSpinBox()
        spacing_spin.setRange(50, 200)
        spacing_spin.setValue(80)
        layout.addRow("Vertical Spacing:", spacing_spin)
        
        neuron_type_combo = QtWidgets.QComboBox()
        neuron_type_combo.addItems(["default", "novelty", "stress", "reward"])
        layout.addRow("Neuron Type:", neuron_type_combo)
        
        # Add color selection for the layer
        color_button = QtWidgets.QPushButton("Select Layer Color")
        layer_color = QtGui.QColor(
            random.randint(100, 240),
            random.randint(100, 240),
            random.randint(100, 240)
        )
        color_button.clicked.connect(lambda: self.select_color_dialog(color_button, layer_color))
        layout.addRow("Layer Color:", color_button)
        
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)
        
        if dialog.exec_():
            # Create the layer
            prefix = prefix_edit.text()
            count = count_spin.value()
            x_pos = x_pos_spin.value()
            y_start = y_start_spin.value()
            spacing = spacing_spin.value()
            neuron_type = neuron_type_combo.currentText()
            
            layer_neurons = []
            for i in range(count):
                name = f"{prefix}{i+1}"
                y_pos = y_start + i * spacing
                
                # Add the neuron with custom color attribute
                self.network.add_neuron(name, 50, (x_pos, y_pos), neuron_type)
                # Store the layer color in neuron attributes
                self.network.neurons[name].attributes['layer_color'] = layer_color.name()
                layer_neurons.append(name)
                
                # Update neuron combo box
                self.update_neuron_combo.addItem(name)
            
            # Store layer for future reference
            if not hasattr(self, 'layers'):
                self.layers = {}
            
            layer_name = f"layer{self.layer_counter}"
            self.layers[layer_name] = {
                'neurons': layer_neurons,
                'color': layer_color.name()
            }
            self.layer_counter += 1
            
            # Update visualization
            self.vis.update_layer_boxes()
            self.vis.update()
            
            self.statusBar().showMessage(f"Added layer with {count} neurons")
    
    def connect_layers_dialog(self):
        """Show dialog to connect two layers."""
        if not hasattr(self, 'layers') or len(self.layers) < 2:
            QtWidgets.QMessageBox.warning(self, "Error", "Need at least two layers to connect")
            return
        
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Connect Layers")
        dialog.setMinimumWidth(300)
        
        layout = QtWidgets.QFormLayout(dialog)
        
        source_combo = QtWidgets.QComboBox()
        source_combo.addItems(self.layers.keys())
        layout.addRow("Source Layer:", source_combo)
        
        target_combo = QtWidgets.QComboBox()
        target_combo.addItems(self.layers.keys())
        layout.addRow("Target Layer:", target_combo)
        
        # Default to connecting consecutive layers
        if self.layer_counter >= 2:
            source_combo.setCurrentText(f"layer{self.layer_counter-2}")
            target_combo.setCurrentText(f"layer{self.layer_counter-1}")
        
        connection_type = QtWidgets.QComboBox()
        connection_type.addItems(["Fully Connected", "One-to-One"])
        layout.addRow("Connection Type:", connection_type)
        
        min_weight_spin = QtWidgets.QDoubleSpinBox()
        min_weight_spin.setRange(-1.0, 1.0)
        min_weight_spin.setSingleStep(0.1)
        min_weight_spin.setValue(-0.5)
        layout.addRow("Min Weight:", min_weight_spin)
        
        max_weight_spin = QtWidgets.QDoubleSpinBox()
        max_weight_spin.setRange(-1.0, 1.0)
        max_weight_spin.setSingleStep(0.1)
        max_weight_spin.setValue(0.5)
        layout.addRow("Max Weight:", max_weight_spin)
        
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)
        
        if dialog.exec_():
            # Connect the layers
            source_layer = self.layers[source_combo.currentText()]
            target_layer = self.layers[target_combo.currentText()]
            conn_type = connection_type.currentText()
            min_weight = min_weight_spin.value()
            max_weight = max_weight_spin.value()
            
            # Extract the neuron lists from the layer data
            if isinstance(source_layer, dict) and 'neurons' in source_layer:
                source_neurons = source_layer['neurons']
            else:
                source_neurons = source_layer  # Handle old format

            if isinstance(target_layer, dict) and 'neurons' in target_layer:
                target_neurons = target_layer['neurons']
            else:
                target_neurons = target_layer  # Handle old format
            
            if conn_type == "Fully Connected":
                # Connect each neuron in source to each in target
                for src in source_neurons:
                    for tgt in target_neurons:
                        weight = random.uniform(min_weight, max_weight)
                        self.network.connect(src, tgt, weight)
                
                self.statusBar().showMessage(f"Connected {len(source_neurons)} source neurons to {len(target_neurons)} target neurons")
            
            elif conn_type == "One-to-One":
                # Connect neurons one-to-one (requires same number of neurons)
                if len(source_neurons) != len(target_neurons):
                    QtWidgets.QMessageBox.warning(self, "Error", "One-to-One connection requires equal number of neurons in both layers")
                    return
                
                for i in range(len(source_neurons)):
                    weight = random.uniform(min_weight, max_weight)
                    self.network.connect(source_neurons[i], target_neurons[i], weight)
                
                self.statusBar().showMessage(f"Connected {len(source_neurons)} neuron pairs one-to-one")
            
            # Update visualization
            self.vis.update()
    
    def create_feedforward_dialog(self):
        """Show dialog to create a feedforward network."""
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Create Feedforward Network")
        dialog.setMinimumWidth(400)
        
        layout = QtWidgets.QVBoxLayout(dialog)
        
        # Layer structure
        structure_group = QtWidgets.QGroupBox("Network Structure")
        structure_layout = QtWidgets.QVBoxLayout(structure_group)
        
        layer_spinners = []
        layer_layout = QtWidgets.QHBoxLayout()
        
        # Add initial 3 layers (input, hidden, output)
        for i, label in enumerate(["Input", "Hidden", "Output"]):
            layer_group = QtWidgets.QGroupBox(label)
            layer_box_layout = QtWidgets.QVBoxLayout(layer_group)
            
            spinner = QtWidgets.QSpinBox()
            spinner.setRange(1, 20)
            spinner.setValue(2 if i == 0 else 3 if i == 1 else 1)
            layer_spinners.append(spinner)
            
            layer_box_layout.addWidget(spinner)
            layer_layout.addWidget(layer_group)
        
        structure_layout.addLayout(layer_layout)
        
        # Add layer button
        add_layer_btn = QtWidgets.QPushButton("Add Hidden Layer")
        add_layer_btn.clicked.connect(lambda: self.add_hidden_layer_spinner(layer_layout, layer_spinners))
        structure_layout.addWidget(add_layer_btn)
        
        layout.addWidget(structure_group)
        
        # Network parameters
        param_group = QtWidgets.QGroupBox("Parameters")
        param_layout = QtWidgets.QFormLayout(param_group)
        
        x_start_spin = QtWidgets.QSpinBox()
        x_start_spin.setRange(50, 1000)
        x_start_spin.setValue(100)
        param_layout.addRow("X Start Position:", x_start_spin)
        
        x_spacing_spin = QtWidgets.QSpinBox()
        x_spacing_spin.setRange(100, 500)
        x_spacing_spin.setValue(200)
        param_layout.addRow("X Spacing:", x_spacing_spin)
        
        y_center_spin = QtWidgets.QSpinBox()
        y_center_spin.setRange(50, 1000)
        y_center_spin.setValue(300)
        param_layout.addRow("Y Center:", y_center_spin)
        
        min_weight_spin = QtWidgets.QDoubleSpinBox()
        min_weight_spin.setRange(-1.0, 1.0)
        min_weight_spin.setSingleStep(0.1)
        min_weight_spin.setValue(-0.5)
        param_layout.addRow("Min Weight:", min_weight_spin)
        
        max_weight_spin = QtWidgets.QDoubleSpinBox()
        max_weight_spin.setRange(-1.0, 1.0)
        max_weight_spin.setSingleStep(0.1)
        max_weight_spin.setValue(0.5)
        param_layout.addRow("Max Weight:", max_weight_spin)
        
        layout.addWidget(param_group)
        
        # Clear existing network checkbox
        clear_check = QtWidgets.QCheckBox("Clear existing network")
        clear_check.setChecked(True)
        layout.addWidget(clear_check)
        
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        
        if dialog.exec_():
            # Get layer sizes
            layer_sizes = [spinner.value() for spinner in layer_spinners]
            
            # Get parameters
            x_start = x_start_spin.value()
            x_spacing = x_spacing_spin.value()
            y_center = y_center_spin.value()
            min_weight = min_weight_spin.value()
            max_weight = max_weight_spin.value()
            
            # Clear existing network if requested
            if clear_check.isChecked():
                self.clear_network()
            
            # Create the feedforward network
            self.create_feedforward_network(layer_sizes, x_start, x_spacing, y_center, min_weight, max_weight)
            
            self.statusBar().showMessage(f"Created feedforward network with {len(layer_sizes)} layers")
    
    def add_hidden_layer_spinner(self, layout, spinners):
        """Add a new hidden layer spinner to the feedforward network dialog."""
        # Create new hidden layer group
        layer_group = QtWidgets.QGroupBox(f"Hidden {len(spinners) - 1}")
        layer_box_layout = QtWidgets.QVBoxLayout(layer_group)
        
        spinner = QtWidgets.QSpinBox()
        spinner.setRange(1, 20)
        spinner.setValue(3)
        
        layer_box_layout.addWidget(spinner)
        
        # Insert before the output layer
        layout.insertWidget(layout.count() - 1, layer_group)
        
        # Insert in spinners list before the output spinner
        spinners.insert(len(spinners) - 1, spinner)
    
    def create_feedforward_network(self, layer_sizes, x_start, x_spacing, y_center, min_weight, max_weight):
        """Create a feedforward neural network with the given configuration."""
        self.layers = {}
        
        # Generate base names for layers
        base_names = ["input"]
        for i in range(1, len(layer_sizes) - 1):
            base_names.append(f"hidden{i}")
        base_names.append("output")
        
        # Create layers
        for layer_idx, (base_name, size) in enumerate(zip(base_names, layer_sizes)):
            layer_neurons = []
            
            for i in range(size):
                name = f"{base_name}_{i+1}"
                
                # Calculate position
                x = x_start + layer_idx * x_spacing
                
                # Calculate y position to center the layer
                total_height = (size - 1) * 80  # 80px spacing between neurons
                y_start = y_center - total_height / 2
                y = y_start + i * 80
                
                # Add the neuron
                self.network.add_neuron(name, 50, (x, y), "default")
                layer_neurons.append(name)
                
                # Update neuron combo box
                self.update_neuron_combo.addItem(name)
            
            # Store layer
            self.layers[f"layer{layer_idx}"] = layer_neurons
        
        # Connect layers
        for i in range(len(layer_sizes) - 1):
            source_layer = self.layers[f"layer{i}"]
            target_layer = self.layers[f"layer{i+1}"]
            
            # Connect each neuron in source to each in target (fully connected)
            for src in source_layer:
                for tgt in target_layer:
                    weight = random.uniform(min_weight, max_weight)
                    self.network.connect(src, tgt, weight)
        
        # Update the layer counter
        self.layer_counter = len(layer_sizes)
        
        # Update visualization
        self.vis.update()
    
    def auto_layout_network(self):
        """Automatically layout the network for better visualization."""
        if not self.network.neurons:
            return
        
        # Simple force-directed layout algorithm
        iterations = 50
        repulsion = 10000  # Strength of repulsion between all neurons
        attraction = 0.1   # Strength of attraction along connections
        
        for _ in range(iterations):
            # Calculate forces
            forces = {name: [0, 0] for name in self.network.neurons}
            
            # Repulsive forces (all neurons repel each other)
            for name1, n1 in self.network.neurons.items():
                pos1 = n1.get_position()
                
                for name2, n2 in self.network.neurons.items():
                    if name1 == name2:
                        continue
                    
                    pos2 = n2.get_position()
                    
                    # Calculate distance and direction
                    dx = pos1[0] - pos2[0]
                    dy = pos1[1] - pos2[1]
                    dist = max(1, (dx**2 + dy**2)**0.5)
                    
                    # Force is inversely proportional to distance squared
                    force = repulsion / (dist**2)
                    
                    # Add force in the direction away from other neuron
                    forces[name1][0] += force * dx / dist
                    forces[name1][1] += force * dy / dist
            
            # Attractive forces (connected neurons attract each other)
            for (src, tgt), conn in self.network.connections.items():
                if src not in self.network.neurons or tgt not in self.network.neurons:
                    continue
                    
                pos1 = self.network.neurons[src].get_position()
                pos2 = self.network.neurons[tgt].get_position()
                
                # Calculate distance and direction
                dx = pos2[0] - pos1[0]
                dy = pos2[1] - pos1[1]
                dist = max(1, (dx**2 + dy**2)**0.5)
                
                # Force is proportional to distance
                force = attraction * dist
                
                # Add force in the direction toward connected neuron
                forces[src][0] += force * dx / dist
                forces[tgt][0] -= force * dx / dist
                forces[src][1] += force * dy / dist
                forces[tgt][1] -= force * dy / dist
            
            # Apply forces
            for name, neuron in self.network.neurons.items():
                pos = neuron.get_position()
                fx, fy = forces[name]
                
                # Limit maximum movement
                max_move = 20
                fx = max(-max_move, min(max_move, fx))
                fy = max(-max_move, min(max_move, fy))
                
                # Update position
                new_x = max(50, min(1150, pos[0] + fx))
                new_y = max(50, min(550, pos[1] + fy))
                neuron.set_position(new_x, new_y)
        
        # Update visualization
        self.vis.update()
        self.statusBar().showMessage("Network layout optimized")
    
    def update_neuron_state(self):
        """Update a specific neuron's state from the simulation controls."""
        neuron = self.update_neuron_combo.currentText()
        if neuron == "Select Neuron":
            QtWidgets.QMessageBox.warning(self, "Error", "Please select a neuron")
            return
        
        value = self.update_value_spin.value()
        
        # Update state
        self.network.update_state({neuron: value})
        
        # Highlight the neuron
        self.vis.highlight_neuron(neuron, 1.0)
        
        # Update visualization
        self.vis.update()
        
        self.statusBar().showMessage(f"Updated {neuron} to value {value}")
    
    def perform_learning(self):
        """Perform Hebbian learning on the network."""
        updated = self.network.perform_learning()
        
        # Update visualization
        self.vis.update()
        
        if updated:
            self.statusBar().showMessage(f"Learning performed: Updated {len(updated)} neuron pairs")
        else:
            self.statusBar().showMessage("Learning performed: No neurons were active enough")
    
    def propagate_activation(self):
        """Propagate activation through the network."""
        self.network.propagate_activation()
        
        # Update visualization
        self.vis.update()
        
        self.statusBar().showMessage("Activation propagated through network")
    
    def trigger_neurogenesis(self):
        """Manually trigger neurogenesis."""
        # Show dialog to configure neurogenesis triggers
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Trigger Neurogenesis")
        dialog.setMinimumWidth(300)
        
        layout = QtWidgets.QFormLayout(dialog)
        
        novelty_spin = QtWidgets.QDoubleSpinBox()
        novelty_spin.setRange(0, 10)
        novelty_spin.setSingleStep(0.1)
        novelty_spin.setValue(5.0)  # High value to ensure triggering
        layout.addRow("Novelty Exposure:", novelty_spin)
        
        stress_spin = QtWidgets.QDoubleSpinBox()
        stress_spin.setRange(0, 2)
        stress_spin.setSingleStep(0.1)
        stress_spin.setValue(1.0)
        layout.addRow("Sustained Stress:", stress_spin)
        
        reward_spin = QtWidgets.QDoubleSpinBox()
        reward_spin.setRange(0, 2)
        reward_spin.setSingleStep(0.1)
        reward_spin.setValue(1.0)
        layout.addRow("Recent Rewards:", reward_spin)
        
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)
        
        if dialog.exec_():
            # Prepare neurogenesis triggers
            state = {
                "novelty_exposure": novelty_spin.value(),
                "sustained_stress": stress_spin.value(),
                "recent_rewards": reward_spin.value()
            }
            
            # Reset cooldown to force neurogenesis
            if hasattr(self.network.neurogenesis, 'neurogenesis_data'):
                self.network.neurogenesis.neurogenesis_data['last_neuron_time'] = 0
            
            # Trigger neurogenesis
            if self.network.check_neurogenesis(state):
                # Get the new neuron name
                new_neuron = self.network.neurogenesis.neurogenesis_data['new_neurons'][-1]
                
                # Highlight the new neuron
                self.vis.highlight_new_neuron(new_neuron, 5.0)
                
                # Update neuron combo box
                self.update_neuron_combo.addItem(new_neuron)
                
                self.statusBar().showMessage(f"Neurogenesis successful: Created {new_neuron}")
            else:
                self.statusBar().showMessage("Neurogenesis failed")
    
    def toggle_auto_simulation(self, state):
        """Toggle auto-simulation mode."""
        if state == QtCore.Qt.Checked:
            # Start auto-simulation timer
            self.auto_sim_timer.start(500)  # Update every 500ms
            self.statusBar().showMessage("Auto-simulation started")
        else:
            # Stop timer
            self.auto_sim_timer.stop()
            self.statusBar().showMessage("Auto-simulation stopped")
    
    def auto_simulation_step(self):
        """Perform one step of auto-simulation."""
        # Randomly stimulate neurons
        for _ in range(2):  # Stimulate 2 random neurons
            if self.network.neurons:
                neuron = random.choice(list(self.network.neurons.keys()))
                value = random.uniform(0, 100)
                self.network.update_state({neuron: value})
                self.vis.highlight_neuron(neuron, 0.5)
        
        # Propagate activation
        self.network.propagate_activation()
        
        # Occasionally perform learning
        if random.random() < 0.2:  # 20% chance
            self.network.perform_learning()
        
        # Very rarely trigger neurogenesis
        if random.random() < 0.05:  # 5% chance
            state = {
                "novelty_exposure": random.uniform(0, 5),
                "sustained_stress": random.uniform(0, 1),
                "recent_rewards": random.uniform(0, 1)
            }
            
            if self.network.check_neurogenesis(state):
                # Get the new neuron name
                new_neuron = self.network.neurogenesis.neurogenesis_data['new_neurons'][-1]
                
                # Highlight the new neuron
                self.vis.highlight_new_neuron(new_neuron, 2.0)
                
                # Update neuron combo box
                self.update_neuron_combo.addItem(new_neuron)
        
        # Update visualization
        self.vis.update()
    
    def new_network(self):
        """Create a new empty network."""
        reply = QtWidgets.QMessageBox.question(
            self, "New Network", 
            "Are you sure you want to create a new network? Unsaved changes will be lost.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
        
        if reply == QtWidgets.QMessageBox.Yes:
            self.clear_network()
            self.statusBar().showMessage("Created new network")
    
    def open_network(self):
        """Open a network from a file."""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Network", "", "JSON Files (*.json)"
        )
        
        if file_path:
            loaded_network = Network.load(file_path)
            if loaded_network:
                self.network = loaded_network
                
                # Update visualization
                self.vis.network = self.network
                self.vis.update()
                
                # Update neuron combo box
                self.update_neuron_combo.clear()
                self.update_neuron_combo.addItem("Select Neuron")
                for name in self.network.neurons:
                    self.update_neuron_combo.addItem(name)
                
                # Reset counters
                self.neuron_counter = len(self.network.neurons)
                self.layer_counter = 0
                
                self.statusBar().showMessage(f"Loaded network from {file_path}")
            else:
                QtWidgets.QMessageBox.warning(self, "Error", "Failed to load network")
    
    def save_network(self):
        """Save the network to a file."""
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Network", "", "JSON Files (*.json)"
        )
        
        if file_path:
            success = self.network.save(file_path)
            if success:
                self.statusBar().showMessage(f"Network saved to {file_path}")
            else:
                QtWidgets.QMessageBox.warning(self, "Error", "Failed to save network")
    
    def clear_network(self):
        """Clear the current network."""
        self.network = Network()
        
        # Update visualization
        self.vis.network = self.network
        self.vis.update()
        
        # Clear neuron combo box
        self.update_neuron_combo.clear()
        self.update_neuron_combo.addItem("Select Neuron")
        
        # Reset counters
        self.neuron_counter = 0
        self.layer_counter = 0
        
        # Clear selection
        self.selected_neuron = None
        self.properties_stack.setCurrentIndex(2)  # Show 'nothing selected'
        
        # Reset layers
        self.layers = {}
    
    def randomize_weights(self):
        """Randomize all connection weights."""
        for connection in self.network.connections.values():
            weight = random.uniform(-1.0, 1.0)
            connection.set_weight(weight)
        
        # Update visualization
        self.vis.update()
        
        self.statusBar().showMessage("All connection weights randomized")
    
    def show_about(self):
        """Show about dialog."""
        QtWidgets.QMessageBox.about(
            self, "About Neural Network Builder",
            "Neural Network Builder\n\n"
            "A visual tool for creating and experimenting with Hebbian neural networks.\n\n"
            "Features:\n"
            "- Visual network design\n"
            "- Hebbian learning\n"
            "- Neurogenesis\n"
            "- Auto-simulation\n\n"
            "Based on the NeuralNetwork library."
        )

class VisualizationEventFilter(QtCore.QObject):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
    
    def eventFilter(self, watched, event):
        if event.type() == QtCore.QEvent.MouseButtonPress:
            self.parent.visualization_mouse_press(event)
            return False  # Allow event to propagate to original handler
        elif event.type() == QtCore.QEvent.MouseMove:
            self.parent.visualization_mouse_move(event)
            return False
        elif event.type() == QtCore.QEvent.MouseButtonRelease:
            self.parent.visualization_mouse_release(event)
            return False
        return False


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = NetworkBuilderGUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    import sys
    from PyQt5 import QtWidgets
    
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")  # More modern style
    
    window = NetworkBuilderGUI()
    window.show()
    
    sys.exit(app.exec_())