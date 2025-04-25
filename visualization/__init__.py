"""
Visualization tools for neural networks.

This module provides GUI components for visualizing and interacting with networks.
Requires PyQt5 to be installed.
"""

try:
    from .network_widget import NetworkVisualization
    from .network_builder_gui import NetworkBuilderGUI
except ImportError:
    import warnings
    warnings.warn("PyQt5 is required for visualization. Install with: pip install PyQt5")
