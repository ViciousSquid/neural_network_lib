"""
Learning algorithms for neural networks.

This module provides various learning mechanisms:
- HebbianLearning: Implements "neurons that fire together, wire together"
- Neurogenesis: Creates new neurons during learning
- BackpropNetwork: Supervised learning with backpropagation
"""

from .hebbian import HebbianLearning
from .neurogenesis import Neurogenesis
from .backprop import BackpropNetwork
