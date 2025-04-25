"""
Neural Network Library

A comprehensive neural network library featuring:
- Core network components (neurons, connections)
- Multiple learning algorithms (Hebbian, neurogenesis, backpropagation)
- Visualization tools with PyQt5

This library provides flexible neural network capabilities with support
for both biologically-inspired learning (Hebbian, neurogenesis) and
traditional machine learning approaches (backpropagation).
"""

__version__ = "0.1.0"

# Import core components for easier access
from .core import Network, Neuron, Connection, Config

# Conditional import for visualization components
try:
    from .visualization import NetworkVisualization, NetworkBuilderGUI
    __has_visualization__ = True
except ImportError:
    __has_visualization__ = False