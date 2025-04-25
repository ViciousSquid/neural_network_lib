"""
Network visualization widget module.

This module provides a PyQt5 widget for visualizing neural networks,
with support for interactive manipulation, zooming, and animation.
"""

from PyQt5 import QtCore, QtGui, QtWidgets
import time
import math

class NetworkVisualization(QtWidgets.QWidget):
    """
    Widget for visualizing neural networks.
    
    Provides a graphical representation of neural networks with interactive
    features for visualization and manipulation. Neurons are displayed as shapes
    with colors based on their type and activation. Connections are shown as
    lines with thickness and color indicating weight.
    
    Signals:
        neuronClicked (str): Emitted when a neuron is clicked
        
    Attributes:
        network: The neural network to visualize
        show_weights (bool): Whether to show connection weights
        show_links (bool): Whether to show connections
        show_labels (bool): Whether to show neuron labels
        highlight_active (bool): Whether to highlight active neurons
        dragging (bool): Whether a neuron is being dragged
        dragged_neuron (str): Name of neuron being dragged
        drag_start_pos (QPoint): Start position of drag
        selection_rect (QRectF): Rectangle for multiple selection
        selected_neurons (set): Set of selected neuron names
        layer_boxes (dict): Layer information for visualization
        zoom_level (float): Current zoom level
        active_highlights (dict): Currently highlighted neurons
        neurogenesis_highlight (dict): Highlight data for new neurons
    """
    
    neuronClicked = QtCore.pyqtSignal(str)
    
    def __init__(self, network):
        """
        Initialize the network visualization widget.
        
        Args:
            network: The neural network to visualize
        """
        super().__init__()
        self.network = network
        self.show_weights = True
        self.show_links = True
        self.show_labels = True
        self.highlight_active = True
        
        # Interaction state
        self.dragging = False
        self.dragged_neuron = None
        self.drag_start_pos = None
        self.setMouseTracking(True)
        
        # Multiple selection state
        self.selection_rect = None
        self.selection_start = None
        self.selected_neurons = set()
        self.is_selecting = False
        
        # Layer visualization
        self.layer_boxes = {}  # Will be populated with layer info from the network
        self.dragging_layer = None
        
        # Visualization style
        self.neuron_radius = 25
        self.font_size = 8
        self.highlight_color = QtGui.QColor(255, 255, 0)
        self.label_width = 150  # Wider label width
        
        # Zooming support
        self.zoom_level = 1.0
        
        # Highlights
        self.neurogenesis_highlight = {
            'neuron': None,
            'start_time': 0,
            'duration': 5.0
        }
        self.active_highlights = {}  # neuron_name -> end_time
        
        # Animation
        self.animation_timer = QtCore.QTimer()
        self.animation_timer.timeout.connect(self.update)
        self.animation_timer.start(50)  # 20 fps
        
        # Update layer information if available
        self.update_layer_boxes()
    
    def update_layer_boxes(self):
        """
        Update layer box information from the network.
        
        Scans the parent GUI for layer information and updates the
        layer boxes used for visualization.
        """
        # Clear existing layer boxes
        self.layer_boxes = {}
        
        # Check if we have layer information in the network
        parent_gui = self.parent()
        while parent_gui and not hasattr(parent_gui, 'layers'):
            parent_gui = parent_gui.parent()
        
        if parent_gui and hasattr(parent_gui, 'layers'):
            for layer_name, layer_data in parent_gui.layers.items():
                # Handle both old and new format of layer data
                if isinstance(layer_data, dict) and 'neurons' in layer_data:
                    neurons = layer_data['neurons']
                else:
                    neurons = layer_data  # Old format
                
                # Calculate bounding box for all neurons in this layer
                if not neurons:
                    continue
                    
                min_x = float('inf')
                min_y = float('inf')
                max_x = float('-inf')
                max_y = float('-inf')
                
                # Find bounds of all neurons in the layer
                for neuron_name in neurons:
                    if neuron_name in self.network.neurons:
                        pos = self.network.neurons[neuron_name].get_position()
                        min_x = min(min_x, pos[0] - self.neuron_radius - 10)
                        min_y = min(min_y, pos[1] - self.neuron_radius - 10)
                        max_x = max(max_x, pos[0] + self.neuron_radius + 10)
                        max_y = max(max_y, pos[1] + self.neuron_radius + 40)  # Extra space for labels
                
                # Add padding
                max_y += 20
                
                # Store layer box info
                if min_x != float('inf'):
                    self.layer_boxes[layer_name] = {
                        'rect': QtCore.QRectF(min_x, min_y, max_x - min_x, max_y - min_y),
                        'neurons': neurons
                    }
    
    def paintEvent(self, event):
        """
        Paint the visualization with zoom support.
        
        Renders all components of the network visualization, including
        connections, neurons, layers, and statistics.
        
        Args:
            event (QPaintEvent): The paint event
        """
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        
        # Apply zoom transformation
        painter.scale(self.zoom_level, self.zoom_level)
        
        # Calculate scale for responsive design
        scale_x = self.width() / (1200 * self.zoom_level)
        scale_y = self.height() / (600 * self.zoom_level)
        scale = min(scale_x, scale_y)
        
        # Fill background
        painter.fillRect(QtCore.QRectF(0, 0, 1200, 600), QtGui.QColor(240, 240, 240))
        
        # Draw connections
        if self.show_links:
            self.draw_connections(painter, scale)
                
        # Draw neurons
        self.draw_neurons(painter, scale)
        
        # Draw layer rectangles if available
        if self.layer_boxes:
            for layer_name, layer_info in self.layer_boxes.items():
                rect = layer_info['rect']
                painter.setPen(QtGui.QPen(QtGui.QColor(100, 100, 200, 100), 2))
                painter.setBrush(QtGui.QBrush(QtGui.QColor(200, 200, 255, 50)))
                painter.drawRect(rect)
                    
                # Draw layer name
                painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 100)))
                font = painter.font()
                font.setBold(True)
                painter.setFont(font)
                painter.drawText(QtCore.QPointF(rect.x() + 10, rect.y() + 20), layer_name)
        
        # Draw highlight for new neurons
        self.draw_highlights(painter, scale)
        
        # Draw the selection rectangle if active
        if self.selection_rect:
            painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 255, 150), 1, QtCore.Qt.DashLine))
            painter.setBrush(QtGui.QBrush(QtGui.QColor(0, 0, 255, 50)))
            painter.drawRect(self.selection_rect)
        
        # Draw network statistics LAST to ensure highest Z-order
        self.draw_statistics(painter, scale)
    
    def draw_connections(self, painter, scale):
        """
        Draw network connections.
        
        Renders the connections between neurons with visual cues for
        weight (thickness, style, color) and optional weight labels.
        
        Args:
            painter (QPainter): The painter object
            scale (float): Display scale factor
        """
        for (source, target), connection in self.network.connections.items():
            # Skip if neuron doesn't exist (safety check)
            if source not in self.network.neurons or target not in self.network.neurons:
                continue
                
            start = self.network.neurons[source].get_position()
            end = self.network.neurons[target].get_position()
            weight = connection.get_weight()
            
            # Determine line color based on weight sign
            if weight > 0:
                color = QtGui.QColor(0, min(255, int(255 * abs(weight))), 0)  # Green for positive
            else:
                color = QtGui.QColor(min(255, int(255 * abs(weight))), 0, 0)  # Red for negative
            
            # Determine line thickness and style based on weight magnitude
            if abs(weight) < 0.1:  # Very weak connection
                pen_style = QtCore.Qt.DotLine
                line_width = 1 * scale
            elif abs(weight) < 0.3:  # Weak connection
                pen_style = QtCore.Qt.DashLine
                line_width = 1 * scale
            elif abs(weight) < 0.6:  # Moderate connection
                pen_style = QtCore.Qt.SolidLine
                line_width = 2 * scale
            else:  # Strong connection
                pen_style = QtCore.Qt.SolidLine
                line_width = 3 * scale
            
            # Create pen with appropriate style and width
            painter.setPen(QtGui.QPen(color, line_width, pen_style))
            painter.drawLine(int(start[0]), int(start[1]), int(end[0]), int(end[1]))
            
            # Add weight text if enabled
            if self.show_weights and abs(weight) > 0.1:
                midpoint = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
                
                # Text background
                text_width = 40
                text_height = 20
                rect = QtCore.QRectF(
                    midpoint[0] - text_width/2,
                    midpoint[1] - text_height/2,
                    text_width, 
                    text_height
                )
                
                painter.setBrush(QtGui.QBrush(QtGui.QColor(0, 0, 0, 180)))
                painter.setPen(QtCore.Qt.NoPen)
                painter.drawRoundedRect(rect, 5, 5)
                
                # Weight text
                font = painter.font()
                font.setPointSize(int(self.font_size * scale))
                painter.setFont(font)
                painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255)))
                painter.drawText(rect, QtCore.Qt.AlignCenter, f"{weight:.2f}")
    
    def draw_neurons(self, painter, scale):
        """
        Draw network neurons with colors based on their layers.
        
        Renders neurons as shapes with colors and sizes based on their
        type, activation, and whether they are in a specific layer.
        
        Args:
            painter (QPainter): The painter object
            scale (float): Display scale factor
        """
        # Check if we have layer information to identify input/output neurons
        input_neurons = set()
        output_neurons = set()
        
        # Try to access layers from the backprop network if available
        if hasattr(self.network, 'backprop') and hasattr(self.network.backprop, 'layers'):
            if self.network.backprop.layers:
                input_neurons = set(self.network.backprop.layers[0])
                output_neurons = set(self.network.backprop.layers[-1])
        
        for name, neuron in self.network.neurons.items():
            pos = neuron.get_position()
            value = self.network.get_neuron_value(name)
            neuron_type = neuron.type
            
            # Check if neuron has a layer color
            layer_color = None
            if 'layer_color' in neuron.attributes:
                layer_color = QtGui.QColor(neuron.attributes['layer_color'])
            
            # Determine if neuron is selected
            is_selected = name in self.selected_neurons
            
            # Determine shape and color based on neuron type and layer
            if name in input_neurons:
                # Input neurons - blue
                self.draw_circular_neuron(painter, pos[0], pos[1], value, name, scale, 
                                        base_color=layer_color or QtGui.QColor(100, 150, 255),
                                        is_selected=is_selected)
            elif name in output_neurons:
                # Output neurons - purple
                self.draw_circular_neuron(painter, pos[0], pos[1], value, name, scale,
                                        base_color=layer_color or QtGui.QColor(200, 100, 255),
                                        is_selected=is_selected)
            elif neuron_type == "default":
                self.draw_circular_neuron(painter, pos[0], pos[1], value, name, scale,
                                        base_color=layer_color,
                                        is_selected=is_selected)
            elif neuron_type in ["novelty", "novel"]:
                self.draw_triangular_neuron(painter, pos[0], pos[1], value, name, scale, 
                                        base_color=layer_color or QtGui.QColor(255, 255, 150),
                                        is_selected=is_selected)
            elif neuron_type in ["stress", "defense"]:
                self.draw_triangular_neuron(painter, pos[0], pos[1], value, name, scale,
                                        base_color=layer_color or QtGui.QColor(255, 150, 150),
                                        is_selected=is_selected)
            elif neuron_type in ["reward"]:
                self.draw_triangular_neuron(painter, pos[0], pos[1], value, name, scale,
                                        base_color=layer_color or QtGui.QColor(150, 255, 150),
                                        is_selected=is_selected)
            else:
                self.draw_square_neuron(painter, pos[0], pos[1], value, name, scale,
                                    base_color=layer_color,
                                    is_selected=is_selected)
    
    def draw_circular_neuron(self, painter, x, y, value, label, scale, base_color=None, is_selected=False):
        """
        Draw a circular neuron.
        
        Renders a neuron as a circle with appropriate colors, labels, and
        selection indicators if selected.
        
        Args:
            painter (QPainter): The painter object
            x, y (float): Position coordinates
            value (float): Neuron value/activation
            label (str): Neuron name
            scale (float): Display scale factor
            base_color (QColor, optional): Optional base color override
            is_selected (bool, optional): Whether this neuron is selected
        """
        # Calculate color based on activation
        intensity = min(255, int((value / 100.0) * 255))
        
        if base_color:
            # Start with the base color and adjust intensity
            r, g, b = base_color.red(), base_color.green(), base_color.blue()
            color = QtGui.QColor(
                min(255, r + intensity//3),
                min(255, g + intensity//3),
                min(255, b + intensity//3)
            )
        else:
            # Default color scheme (blue)
            color = QtGui.QColor(255 - intensity, 255 - intensity + intensity//2, 255)
        
        # Draw neuron circle
        painter.setBrush(QtGui.QBrush(color))
        painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0)))
        radius = int(self.neuron_radius * scale)
        painter.drawEllipse(
            int(x - radius), 
            int(y - radius),
            int(radius * 2), 
            int(radius * 2)
        )
        
        # Draw selection indicator
        if is_selected:
            painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 255), 2 * scale))
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.drawEllipse(
                int(x - radius - 5 * scale),
                int(y - radius - 5 * scale),
                int((radius + 5) * 2 * scale),
                int((radius + 5) * 2 * scale)
            )
        
        # Draw label
        if self.show_labels:
            font = painter.font()
            font.setPointSize(int(self.font_size * scale))
            painter.setFont(font)
            painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0)))
            
            # Label background
            label_width = int(self.label_width * scale)
            label_height = int(20 * scale)
            label_rect = QtCore.QRectF(
                int(x - label_width/2),
                int(y + radius + 5),
                label_width,
                label_height
            )
            
            # Add value if showing activation
            if self.highlight_active:
                display_text = f"{label}\n{value:.1f}"
            else:
                display_text = label
                
            painter.drawText(label_rect, QtCore.Qt.AlignCenter, display_text)
    
    def draw_square_neuron(self, painter, x, y, value, label, scale, base_color=None, is_selected=False):
        """
        Draw a square neuron.
        
        Renders a neuron as a square with appropriate colors and labels.
        
        Args:
            painter (QPainter): The painter object
            x, y (float): Position coordinates
            value (float): Neuron value/activation
            label (str): Neuron name
            scale (float): Display scale factor
            base_color (QColor, optional): Optional base color override
            is_selected (bool, optional): Whether this neuron is selected
        """
        # Calculate color based on activation
        intensity = min(255, int((value / 100.0) * 255))
        
        if base_color:
            # Start with the base color and adjust intensity
            r, g, b = base_color.red(), base_color.green(), base_color.blue()
            color = QtGui.QColor(
                min(255, r + intensity//3),
                min(255, g + intensity//3),
                min(255, b + intensity//3)
            )
        else:
            # Default color scheme (green)
            color = QtGui.QColor(255 - intensity, 255, 255 - intensity)
        
        # Draw neuron square
        painter.setBrush(QtGui.QBrush(color))
        painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0)))
        side = int(self.neuron_radius * 1.8 * scale)
        painter.drawRect(
            int(x - side/2),
            int(y - side/2),
            side,
            side
        )
        
        # Draw selection indicator
        if is_selected:
            painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 255), 2 * scale))
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.drawRect(
                int(x - side/2 - 5 * scale),
                int(y - side/2 - 5 * scale),
                int(side + 10 * scale),
                int(side + 10 * scale)
            )
        
        # Draw label
        if self.show_labels:
            font = painter.font()
            font.setPointSize(int(self.font_size * scale))
            painter.setFont(font)
            painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0)))
            
            label_width = int(self.label_width * scale)
            label_height = int(20 * scale)
            label_rect = QtCore.QRectF(
                int(x - label_width/2),
                int(y + side/2 + 5),
                label_width,
                label_height
            )
            
            # Add value if showing activation
            if self.highlight_active:
                display_text = f"{label}\n{value:.1f}"
            else:
                display_text = label
                
            painter.drawText(label_rect, QtCore.Qt.AlignCenter, display_text)
    
    def draw_triangular_neuron(self, painter, x, y, value, label, scale, base_color=None, is_selected=False):
        """
        Draw a triangular neuron.
        
        Renders a neuron as a triangle with appropriate colors and labels.
        Used for special neuron types like novelty, stress, or reward.
        
        Args:
            painter (QPainter): The painter object
            x, y (float): Position coordinates
            value (float): Neuron value/activation
            label (str): Neuron name
            scale (float): Display scale factor
            base_color (QColor, optional): Optional base color override
            is_selected (bool, optional): Whether this neuron is selected
        """
        # Calculate color based on activation
        intensity = min(255, int((value / 100.0) * 255))
        
        if base_color:
            # Adjust provided base color based on intensity
            r, g, b = base_color.red(), base_color.green(), base_color.blue()
            color = QtGui.QColor(
                min(255, r + intensity//3),
                min(255, g + intensity//3),
                min(255, b + intensity//3)
            )
        else:
            # Default color scheme (orange)
            color = QtGui.QColor(255, 200 + intensity//4, 100 + intensity//2)
        
        # Create triangle shape
        triangle = QtGui.QPolygonF()
        size = int(self.neuron_radius * 1.5 * scale)
        
        triangle.append(QtCore.QPointF(x, y - size))  # Top
        triangle.append(QtCore.QPointF(x - size, y + size))  # Bottom left
        triangle.append(QtCore.QPointF(x + size, y + size))  # Bottom right
        
        # Draw the triangle
        painter.setBrush(QtGui.QBrush(color))
        painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0)))
        painter.drawPolygon(triangle)
        
        # Draw selection indicator
        if is_selected:
            painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 255), 2 * scale))
            painter.setBrush(QtCore.Qt.NoBrush)
            
            # Bigger triangle for selection indicator
            sel_triangle = QtGui.QPolygonF()
            sel_size = int((self.neuron_radius + 5) * 1.5 * scale)
            
            sel_triangle.append(QtCore.QPointF(x, y - sel_size))  # Top
            sel_triangle.append(QtCore.QPointF(x - sel_size, y + sel_size))  # Bottom left
            sel_triangle.append(QtCore.QPointF(x + sel_size, y + sel_size))  # Bottom right
            
            painter.drawPolygon(sel_triangle)
        
        # Draw label
        if self.show_labels:
            font = painter.font()
            font.setPointSize(int(self.font_size * scale))
            painter.setFont(font)
            painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0)))
            
            label_width = int(self.label_width * scale)
            label_height = int(20 * scale)
            label_rect = QtCore.QRectF(
                int(x - label_width/2),
                int(y + size + 5),
                label_width,
                label_height
            )
            
            # Add value if showing activation
            if self.highlight_active:
                display_text = f"{label}\n{value:.1f}"
            else:
                display_text = label
                
            painter.drawText(label_rect, QtCore.Qt.AlignCenter, display_text)
    
    def draw_highlights(self, painter, scale):
        """
        Draw highlights for new and active neurons.
        
        Renders visual highlights for newly created neurons (from neurogenesis)
        and for neurons that have recently been activated.
        
        Args:
            painter (QPainter): The painter object
            scale (float): Display scale factor
        """
        current_time = time.time()
        
        # Highlight for neurogenesis
        if (self.neurogenesis_highlight['neuron'] and 
            current_time - self.neurogenesis_highlight['start_time'] < self.neurogenesis_highlight['duration']):
            
            neuron_name = self.neurogenesis_highlight['neuron']
            if neuron_name in self.network.neurons:
                pos = self.network.neurons[neuron_name].get_position()
                
                # Pulsing effect
                elapsed = current_time - self.neurogenesis_highlight['start_time']
                pulse = 1 + 0.3 * math.sin(elapsed * 5)  # Pulsing factor
                
                # Draw highlight circle
                painter.setPen(QtGui.QPen(self.highlight_color, 3 * scale))
                painter.setBrush(QtCore.Qt.NoBrush)
                radius = int(self.neuron_radius * 2 * pulse * scale)
                painter.drawEllipse(
                    int(pos[0] - radius), 
                    int(pos[1] - radius),
                    int(radius * 2), 
                    int(radius * 2)
                )
        
        # Highlights for active neurons
        if self.highlight_active:
            # First clean up expired highlights
            self.active_highlights = {
                name: end_time for name, end_time in self.active_highlights.items()
                if end_time > current_time
            }
            
            # Draw remaining highlights
            for neuron_name, end_time in self.active_highlights.items():
                if neuron_name in self.network.neurons:
                    pos = self.network.neurons[neuron_name].get_position()
                    
                    # Fade out as time elapses
                    time_left = end_time - current_time
                    alpha = min(255, int(255 * time_left))
                    
                    # Draw fade-out glow
                    glow_color = QtGui.QColor(255, 255, 0, alpha)
                    painter.setPen(QtGui.QPen(glow_color, 2 * scale))
                    painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 0, alpha // 3)))
                    
                    radius = int(self.neuron_radius * 1.5 * scale)
                    painter.drawEllipse(
                        int(pos[0] - radius), 
                        int(pos[1] - radius),
                        int(radius * 2), 
                        int(radius * 2)
                    )
    
    def draw_statistics(self, painter, scale):
        """
        Draw network statistics with fixed width and size.
        
        Renders a statistics panel with information about the network,
        such as neuron count, connection count, and average weight.
        
        Args:
            painter (QPainter): The painter object
            scale (float): Display scale factor
        """
        stats = self.network.get_network_statistics()
        
        # Format text
        text = (
            f"Neurons: {stats['neurons']}  |  "
            f"Connections: {stats['connections']}  |  "
            f"Avg Weight: {stats['avg_weight']:.3f}"
        )
        
        # Draw background - right aligned, fixed size regardless of zoom
        stat_rect = QtCore.QRectF(
            1200 - 500,  # Fixed position
            10, 
            480,  # Wider to fit all text
            30
        )
        
        # Save current transform
        painter.save()
        
        # Reset transform to draw at fixed size
        painter.resetTransform()
        
        # Clear any clipping to ensure it's on top
        painter.setClipRect(painter.window(), QtCore.Qt.NoClip)
        
        # Draw at screen coordinates
        screen_rect = QtCore.QRectF(
            self.width() - 500 * scale,
            10 * scale,
            480 * scale,
            30 * scale
        )
        
        painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 255, 200)))
        painter.setPen(QtCore.Qt.NoPen)
        painter.drawRoundedRect(screen_rect, 5, 5)
        
        # Draw text with smaller fixed font
        font = painter.font()
        font.setPointSize(10)  # Smaller fixed size
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0)))
        painter.drawText(screen_rect, QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter, text)
        
        # Restore transform
        painter.restore()
    
    def highlight_neuron(self, neuron_name, duration=1.0):
        """
        Highlight a neuron.
        
        Adds a temporary visual highlight to a neuron to draw attention to it.
        
        Args:
            neuron_name (str): Name of neuron to highlight
            duration (float, optional): Duration in seconds. Defaults to 1.0.
        """
        self.active_highlights[neuron_name] = time.time() + duration
    
    def highlight_new_neuron(self, neuron_name, duration=5.0):
        """
        Highlight a newly created neuron.
        
        Adds a special pulsing highlight to a newly created neuron,
        typically used after neurogenesis.
        
        Args:
            neuron_name (str): Name of neuron to highlight
            duration (float, optional): Duration in seconds. Defaults to 5.0.
        """
        self.neurogenesis_highlight = {
            'neuron': neuron_name,
            'start_time': time.time(),
            'duration': duration
        }
    
    def toggle_weights(self, show=None):
        """
        Toggle display of connection weights.
        
        Controls whether weight values are shown on connections.
        
        Args:
            show (bool, optional): If provided, set state directly. Otherwise toggle.
        """
        if show is not None:
            self.show_weights = show
        else:
            self.show_weights = not self.show_weights
            
        self.update()
    
    def toggle_links(self, show=None):
        """
        Toggle display of connections.
        
        Controls whether connections between neurons are visible.
        
        Args:
            show (bool, optional): If provided, set state directly. Otherwise toggle.
        """
        if show is not None:
            self.show_links = show
        else:
            self.show_links = not self.show_links
            
        self.update()
    
    def toggle_labels(self, show=None):
        """
        Toggle display of neuron labels.
        
        Controls whether neuron names and values are shown.
        
        Args:
            show (bool, optional): If provided, set state directly. Otherwise toggle.
        """
        if show is not None:
            self.show_labels = show
        else:
            self.show_labels = not self.show_labels
            
        self.update()
    
    def toggle_active_highlight(self, show=None):
        """
        Toggle highlighting of active neurons.
        
        Controls whether neurons with high activation are automatically highlighted.
        
        Args:
            show (bool, optional): If provided, set state directly. Otherwise toggle.
        """
        if show is not None:
            self.highlight_active = show
        else:
            self.highlight_active = not self.highlight_active
            
        self.update()

    def show_neuron_tooltip(self, neuron_name, pos):
        """
        Show tooltip with detailed neuron information.
        
        Displays a tooltip with information about a neuron when hovering over it.
        
        Args:
            neuron_name (str): Name of the neuron
            pos (QPoint): Position to show the tooltip
        """
        if neuron_name in self.network.neurons:
            neuron = self.network.neurons[neuron_name]
            value = self.network.get_neuron_value(neuron_name)
            position = neuron.get_position()
            
            # Count incoming and outgoing connections
            incoming = sum(1 for src, tgt in self.network.connections if tgt == neuron_name)
            outgoing = sum(1 for src, tgt in self.network.connections if src == neuron_name)
            
            tooltip_text = f"Name: {neuron_name}\n"
            tooltip_text += f"Type: {neuron.type}\n"
            tooltip_text += f"Value: {value:.2f}\n"
            tooltip_text += f"Position: ({position[0]:.1f}, {position[1]:.1f})\n"
            tooltip_text += f"Connections: {incoming} in, {outgoing} out"
            
            QtWidgets.QToolTip.showText(
                self.mapToGlobal(pos),
                tooltip_text,
                self,
                QtCore.QRect(pos.x()-5, pos.y()-5, 10, 10),
                2000  # Hide after 2 seconds
            )
    
    def select_color_dialog(self, button, color_var):
        """
        Open color picker dialog and update the button color.
        
        Args:
            button (QPushButton): Button to update
            color_var (QColor): Color variable to modify
        """
        color = QtWidgets.QColorDialog.getColor(color_var, self, "Select Layer Color")
        
        if color.isValid():
            # Update the reference variable
            color_var.setRgb(color.red(), color.green(), color.blue())
            
            # Set button background color
            button.setStyleSheet(f"background-color: {color.name()}")
    
    def mousePressEvent(self, event):
        """
        Handle mouse press event.
        
        Manages selection, dragging, and other mouse interactions.
        
        Args:
            event (QMouseEvent): Mouse event
        """
        if event.button() == QtCore.Qt.LeftButton:
            # Check if we're clicking on a layer first
            layer_name = self.check_layer_click(event.pos())
            if layer_name:
                self.dragging_layer = layer_name
                self.drag_start_pos = event.pos()
                return  # Important - return early to prevent other handling
                    
            # Then check if we're clicking on an existing selected neuron
            neuron_clicked = self.check_neuron_click(event.pos())
            if neuron_clicked and neuron_clicked in self.selected_neurons:
                # Start dragging the selected group
                self.dragging = True
                self.dragged_neuron = None  # Not dragging a specific neuron
                self.drag_start_pos = event.pos()
                return
                    
            # Then check if we're clicking on a single neuron
            if neuron_clicked:
                # If shift is pressed, add to selection
                modifiers = QtWidgets.QApplication.keyboardModifiers()
                if modifiers == QtCore.Qt.ShiftModifier:
                    self.selected_neurons.add(neuron_clicked)
                else:
                    # Clear previous selection and select this neuron
                    self.selected_neurons = {neuron_clicked}
                        
                # Start dragging this neuron
                self.dragging = True
                self.dragged_neuron = neuron_clicked
                self.drag_start_pos = event.pos()
                self.neuronClicked.emit(neuron_clicked)
                    
                # Show tooltip with neuron information
                self.show_neuron_tooltip(neuron_clicked, event.pos())
                return
                    
            # If we didn't click on anything, start selection rectangle
            self.selection_start = event.pos()
            self.is_selecting = True
            self.selection_rect = QtCore.QRectF(
                self.selection_start.x() / self.zoom_level, 
                self.selection_start.y() / self.zoom_level,
                0, 0
            )
                
            # Clear selection if not using shift
            modifiers = QtWidgets.QApplication.keyboardModifiers()
            if modifiers != QtCore.Qt.ShiftModifier:
                self.selected_neurons.clear()
    
    def mouseMoveEvent(self, event):
        """
        Handle mouse move event.
        
        Manages dragging of neurons, layers, and selection rectangle.
        
        Args:
            event (QMouseEvent): Mouse event
        """
        if self.dragging_layer and self.drag_start_pos:
            # Calculate drag delta adjusted for zoom
            delta = event.pos() - self.drag_start_pos
            delta_x = delta.x() / self.zoom_level
            delta_y = delta.y() / self.zoom_level
            
            # Move all neurons in the layer
            layer_info = self.layer_boxes[self.dragging_layer]
            for neuron_name in layer_info['neurons']:
                if neuron_name in self.network.neurons:
                    neuron = self.network.neurons[neuron_name]
                    old_pos = neuron.get_position()
                    neuron.set_position(
                        old_pos[0] + delta_x,
                        old_pos[1] + delta_y
                    )
            
            # Update the layer box position
            layer_info['rect'].translate(delta_x, delta_y)
            
            # Update drag start position
            self.drag_start_pos = event.pos()
            self.update()
            return
        
        elif self.dragging and self.drag_start_pos:
            # Calculate drag delta adjusted for zoom
            delta = event.pos() - self.drag_start_pos
            delta_x = delta.x() / self.zoom_level
            delta_y = delta.y() / self.zoom_level
            
            # If we're dragging a specific neuron
            if self.dragged_neuron:
                neuron = self.network.neurons[self.dragged_neuron]
                old_pos = neuron.get_position()
                neuron.set_position(
                    old_pos[0] + delta_x,
                    old_pos[1] + delta_y
                )
            
            # If we're dragging multiple neurons
            for neuron_name in self.selected_neurons:
                if neuron_name == self.dragged_neuron:
                    continue  # Skip if we already moved it
                
                if neuron_name in self.network.neurons:
                    neuron = self.network.neurons[neuron_name]
                    old_pos = neuron.get_position()
                    neuron.set_position(
                        old_pos[0] + delta_x,
                        old_pos[1] + delta_y
                    )
            
            # Update drag start position
            self.drag_start_pos = event.pos()
            self.update()
            return
        
        elif self.is_selecting and self.selection_start:
            # Update selection rectangle
            self.selection_rect = QtCore.QRectF(
                min(self.selection_start.x(), event.x()) / self.zoom_level,
                min(self.selection_start.y(), event.y()) / self.zoom_level,
                abs(event.x() - self.selection_start.x()) / self.zoom_level,
                abs(event.y() - self.selection_start.y()) / self.zoom_level
            )
            
            # Find neurons in the selection rectangle
            for name, neuron in self.network.neurons.items():
                pos = neuron.get_position()
                if self.selection_rect.contains(pos[0], pos[1]):
                    self.selected_neurons.add(name)
            
            self.update()
    
    def mouseReleaseEvent(self, event):
        """
        Handle mouse release event.
        
        Finalizes dragging and selection operations.
        
        Args:
            event (QMouseEvent): Mouse event
        """
        if event.button() == QtCore.Qt.LeftButton:
            self.dragging = False
            self.dragged_neuron = None
            self.dragging_layer = None
            self.is_selecting = False
            self.selection_rect = None
            
            # Update layer boxes since neurons might have moved
            self.update_layer_boxes()
            
            self.update()
    
    def check_neuron_click(self, pos):
        """
        Check if a neuron was clicked.
        
        Determines if a mouse click position is over a neuron.
        
        Args:
            pos (QPoint): Click position
            
        Returns:
            str: Name of clicked neuron or None if no neuron was clicked
        """
        # Convert position to logical coordinates
        logical_x = pos.x() / self.zoom_level
        logical_y = pos.y() / self.zoom_level
        
        # Check each neuron
        for name, neuron in self.network.neurons.items():
            neuron_pos = neuron.get_position()
            
            # Check if click is within neuron radius
            distance = math.sqrt(
                (neuron_pos[0] - logical_x) ** 2 + 
                (neuron_pos[1] - logical_y) ** 2
            )
            
            if distance <= self.neuron_radius:
                return name
                
        return None
    
    def check_layer_click(self, pos):
        """
        Check if click is on a layer box.
        
        Determines if a mouse click position is over a layer box.
        
        Args:
            pos (QPoint): Click position
            
        Returns:
            str: Name of clicked layer or None if no layer was clicked
        """
        scaled_pos = QtCore.QPointF(pos.x() / self.zoom_level, pos.y() / self.zoom_level)
        
        for layer_name, layer_info in self.layer_boxes.items():
            if layer_info['rect'].contains(scaled_pos):
                return layer_name
        return None
    
    def set_zoom(self, zoom_factor):
        """
        Set the zoom level.
        
        Changes the visual zoom level of the network visualization.
        
        Args:
            zoom_factor (float): New zoom level (1.0 = 100%)
        """
        self.zoom_level = zoom_factor
        self.update()