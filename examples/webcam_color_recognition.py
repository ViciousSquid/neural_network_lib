# webcam_color_recognition.py
import sys
import os
import cv2
import numpy as np
import time
from PyQt5 import QtWidgets, QtCore, QtGui

# Ensure neural network library is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from NeuralNetwork.core import Network, Config, Neuron, Connection

class WebcamColorRecognition(QtWidgets.QMainWindow):
    """
    Application that uses a neural network to recognize colors from webcam input.
    Demonstrates using the neural network framework as an API.
    """
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.setup_network()
        self.setup_webcam()
        self.start_processing()
    
    def setup_ui(self):
        """Set up the application UI."""
        self.setWindowTitle("Neural Network Color Recognition")
        self.setGeometry(100, 100, 1000, 600)
        
        # Central widget with layout
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        main_layout = QtWidgets.QHBoxLayout(self.central_widget)
        
        # Left panel for webcam feed
        self.webcam_panel = QtWidgets.QLabel()
        self.webcam_panel.setMinimumSize(640, 480)
        self.webcam_panel.setAlignment(QtCore.Qt.AlignCenter)
        self.webcam_panel.setStyleSheet("background-color: #000;")
        
        # Right panel for network visualization and controls
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        
        # Network visualization
        from NeuralNetwork.visualization import NetworkVisualization
        self.vis = NetworkVisualization(self.network)
        self.vis.setMinimumSize(300, 300)
        right_layout.addWidget(self.vis)
        
        # Controls
        controls_group = QtWidgets.QGroupBox("Controls")
        controls_layout = QtWidgets.QVBoxLayout(controls_group)
        
        # Add sample color button
        self.sample_btn = QtWidgets.QPushButton("Sample Current Color")
        self.sample_btn.clicked.connect(self.sample_current_color)
        controls_layout.addWidget(self.sample_btn)
        
        # Color selection for training
        color_layout = QtWidgets.QHBoxLayout()
        color_layout.addWidget(QtWidgets.QLabel("Assign to:"))
        self.color_combo = QtWidgets.QComboBox()
        self.color_combo.addItems(["Red", "Green", "Blue", "Yellow"])
        color_layout.addWidget(self.color_combo)
        controls_layout.addLayout(color_layout)
        
        # Train button
        self.train_btn = QtWidgets.QPushButton("Train Network")
        self.train_btn.clicked.connect(self.train_network)
        controls_layout.addWidget(self.train_btn)
        
        # Reset button
        self.reset_btn = QtWidgets.QPushButton("Reset Network")
        self.reset_btn.clicked.connect(self.reset_network)
        controls_layout.addWidget(self.reset_btn)
        
        # Recognition result display
        result_layout = QtWidgets.QHBoxLayout()
        result_layout.addWidget(QtWidgets.QLabel("Recognized Color:"))
        self.result_label = QtWidgets.QLabel("None")
        self.result_label.setStyleSheet("font-weight: bold; font-size: 16px;")
        result_layout.addWidget(self.result_label)
        controls_layout.addLayout(result_layout)
        
        # Sample RGB values
        self.rgb_label = QtWidgets.QLabel("RGB: (0, 0, 0)")
        controls_layout.addWidget(self.rgb_label)
        
        # Neuron activation display
        self.activation_labels = {}
        activation_group = QtWidgets.QGroupBox("Output Neuron Activation")
        activation_layout = QtWidgets.QVBoxLayout(activation_group)
        
        for color in ["Red", "Green", "Blue", "Yellow"]:
            layout = QtWidgets.QHBoxLayout()
            layout.addWidget(QtWidgets.QLabel(f"{color}:"))
            label = QtWidgets.QLabel("0%")
            layout.addWidget(label)
            self.activation_labels[color.lower()] = label
            activation_layout.addLayout(layout)
        
        controls_layout.addWidget(activation_group)
        
        # Add controls to right panel
        right_layout.addWidget(controls_group)
        
        # Add panels to main layout
        main_layout.addWidget(self.webcam_panel)
        main_layout.addWidget(right_panel)
        
        # Set up status bar
        self.statusBar().showMessage("Ready")
    
    def setup_network(self):
        """Set up the neural network for color recognition."""
        self.network = Network()
        
        # Create a simple 3-5-4 network (RGB input -> hidden -> color output)
        
        # Input layer (RGB values)
        self.network.add_neuron("red_in", 0, (100, 100), "default")
        self.network.add_neuron("green_in", 0, (100, 200), "default")
        self.network.add_neuron("blue_in", 0, (100, 300), "default")
        
        # Hidden layer
        for i in range(5):
            self.network.add_neuron(f"hidden_{i+1}", 0, (250, 100 + i*50), "default")
        
        # Output layer (color recognition)
        self.network.add_neuron("red_out", 0, (400, 100), "default")
        self.network.add_neuron("green_out", 0, (400, 200), "default")
        self.network.add_neuron("blue_out", 0, (400, 300), "default")
        self.network.add_neuron("yellow_out", 0, (400, 400), "default")
        
        # Connect input to hidden
        for i in range(5):
            self.network.connect("red_in", f"hidden_{i+1}", 0.1)
            self.network.connect("green_in", f"hidden_{i+1}", 0.1)
            self.network.connect("blue_in", f"hidden_{i+1}", 0.1)
        
        # Connect hidden to output
        for i in range(5):
            self.network.connect(f"hidden_{i+1}", "red_out", 0.1)
            self.network.connect(f"hidden_{i+1}", "green_out", 0.1)
            self.network.connect(f"hidden_{i+1}", "blue_out", 0.1)
            self.network.connect(f"hidden_{i+1}", "yellow_out", 0.1)
        
        # Initialize learning modules
        self.network.initialize_learning()
        
        # Set up backpropagation for training
        from backprop_module import BackpropNetwork
        self.backprop = BackpropNetwork(self.network)
        
        input_layer = ["red_in", "green_in", "blue_in"]
        hidden_layer = [f"hidden_{i+1}" for i in range(5)]
        output_layer = ["red_out", "green_out", "blue_out", "yellow_out"]
        
        self.backprop.set_layers([input_layer, hidden_layer, output_layer])
        self.backprop.learning_rate = 0.2
        self.backprop.momentum = 0.9
        
        # Sample data for initial training
        self.training_data = []
    
    def setup_webcam(self):
        """Set up webcam capture."""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QtWidgets.QMessageBox.warning(self, "Error", "Could not open webcam")
        
        # Set up timer for frame processing
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.process_frame)
        
        # Store current color sample
        self.current_rgb = (0, 0, 0)
    
    def start_processing(self):
        """Start webcam frame processing."""
        self.timer.start(50)  # Process frames every 50ms (20 fps)
    
    def process_frame(self):
        """Process webcam frame and update UI."""
        ret, frame = self.cap.read()
        if not ret:
            return
        
        # Get center region color
        height, width, _ = frame.shape
        center_x, center_y = width // 2, height // 2
        center_size = 20
        
        # Draw targeting rectangle
        cv2.rectangle(
            frame,
            (center_x - center_size, center_y - center_size),
            (center_x + center_size, center_y + center_size),
            (0, 255, 0), 2
        )
        
        # Extract RGB values from center region
        center_region = frame[
            center_y - center_size:center_y + center_size,
            center_x - center_size:center_x + center_size
        ]
        
        bgr = center_region.mean(axis=(0, 1))
        # OpenCV uses BGR, convert to RGB
        rgb = (int(bgr[2]), int(bgr[1]), int(bgr[0]))
        self.current_rgb = rgb
        
        # Update RGB label
        self.rgb_label.setText(f"RGB: {rgb}")
        
        # Feed values to network
        r = (rgb[0] / 255) * 100
        g = (rgb[1] / 255) * 100
        b = (rgb[2] / 255) * 100
        
        self.network.update_state({
            "red_in": r,
            "green_in": g,
            "blue_in": b
        })
        
        # Propagate activation
        self.network.propagate_activation()
        
        # Read output activations
        color_outputs = {
            "red": self.network.get_neuron_value("red_out"),
            "green": self.network.get_neuron_value("green_out"),
            "blue": self.network.get_neuron_value("blue_out"),
            "yellow": self.network.get_neuron_value("yellow_out")
        }
        
        # Update activation labels
        for color, value in color_outputs.items():
            self.activation_labels[color].setText(f"{value:.1f}%")
        
        # Determine recognized color
        recognized = max(color_outputs.items(), key=lambda x: x[1])[0]
        
        # Only show as recognized if activation is above threshold
        if color_outputs[recognized] > 60:
            self.result_label.setText(recognized.capitalize())
            
            # Set result label background to recognized color
            if recognized == "red":
                color = "#ffaaaa"
            elif recognized == "green":
                color = "#aaffaa"
            elif recognized == "blue":
                color = "#aaaaff"
            elif recognized == "yellow":
                color = "#ffffaa"
            else:
                color = "#ffffff"
                
            self.result_label.setStyleSheet(f"background-color: {color}; font-weight: bold; font-size: 16px; padding: 5px;")
        else:
            self.result_label.setText("Uncertain")
            self.result_label.setStyleSheet("font-weight: bold; font-size: 16px;")
        
        # Draw color rectangle on frame
        color_rect = np.zeros((100, 100, 3), dtype=np.uint8)
        color_rect[:] = (rgb[2], rgb[1], rgb[0])  # BGR for OpenCV
        
        # Add color rect to frame
        frame[10:110, width-110:width-10] = color_rect
        
        # Update network visualization
        self.vis.update()
        
        # Convert to Qt format for display
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        image = QtGui.QImage(rgb_frame.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(image)
        
        # Resize to fit panel if needed
        pixmap = pixmap.scaled(
            self.webcam_panel.width(), 
            self.webcam_panel.height(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation
        )
        
        self.webcam_panel.setPixmap(pixmap)
    
    def sample_current_color(self):
        """Sample the current color for training."""
        color = self.color_combo.currentText().lower()
        rgb = self.current_rgb
        
        # Normalize to 0-1 range for training
        normalized = [rgb[0]/255, rgb[1]/255, rgb[2]/255]
        
        # Create target output (one-hot encoding)
        target = [0, 0, 0, 0]
        if color == "red":
            target[0] = 1
        elif color == "green":
            target[1] = 1
        elif color == "blue":
            target[2] = 1
        elif color == "yellow":
            target[3] = 1
        
        # Add to training data
        self.training_data.append([normalized, target])
        
        self.statusBar().showMessage(f"Added {color} sample with RGB {rgb}")
    
    def train_network(self):
        """Train the network with collected samples."""
        if not self.training_data:
            QtWidgets.QMessageBox.warning(self, "Error", "No training samples collected")
            return
        
        # Disable UI during training
        self.setEnabled(False)
        
        # Progress dialog
        progress = QtWidgets.QProgressDialog("Training network...", "Cancel", 0, 100, self)
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.show()
        
        # Training callback
        def training_callback(epoch, error):
            progress.setValue(int(epoch / 100 * 100))
            QtWidgets.QApplication.processEvents()
            return not progress.wasCanceled()
        
        try:
            # Train network
            errors = self.backprop.train(
                self.training_data,
                epochs=100,
                target_error=0.01,
                callback=training_callback
            )
            
            final_error = errors[-1] if errors else 0
            
            if progress.wasCanceled():
                self.statusBar().showMessage("Training canceled")
            else:
                progress.setValue(100)
                self.statusBar().showMessage(
                    f"Training complete with {len(self.training_data)} samples. Final error: {final_error:.4f}"
                )
        
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", f"Training failed: {str(e)}")
        
        finally:
            # Re-enable UI
            self.setEnabled(True)
    
    def reset_network(self):
        """Reset the network weights and training data."""
        reply = QtWidgets.QMessageBox.question(
            self, "Reset Network", 
            "Are you sure you want to reset the network? All training will be lost.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
        
        if reply == QtWidgets.QMessageBox.Yes:
            # Reinitialize network
            self.setup_network()
            
            # Update visualization
            self.vis.network = self.network
            self.vis.update()
            
            self.statusBar().showMessage("Network reset to initial state")
    
    def closeEvent(self, event):
        """Handle close event to clean up resources."""
        # Stop webcam processing
        self.timer.stop()
        
        # Release webcam
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        
        event.accept()

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = WebcamColorRecognition()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()