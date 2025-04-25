"""
Core components of the neural network library.

This module provides the fundamental building blocks:
- Neuron: Basic computational unit
- Connection: Links between neurons with weights
- Network: Container for neurons and connections
- Config: Configuration parameters
"""

from .neuron import Neuron
from .connection import Connection
from .network import Network
from .config import Config
