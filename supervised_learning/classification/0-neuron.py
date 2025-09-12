#!/usr/bin/env python3
"""
Neuron class for binary classification
"""

import numpy as np


class Neuron:
    """
    A class that defines a single neuron performing binary classification
    """
    
    def __init__(self, nx):
        """
        Initialize the neuron
        
        Args:
            nx (int): number of input features to the neuron
            
        Raises:
            TypeError: if nx is not an integer
            ValueError: if nx is less than 1
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
            
        # Initialize weights with random normal distribution
        self.__W = np.random.normal(size=(1, nx))
        
        # Initialize bias to 0
        self.__b = 0
        
        # Initialize activated output to 0
        self.__A = 0
    
    @property
    def W(self):
        """Getter for weights"""
        return self.__W
    
    @property
    def b(self):
        """Getter for bias"""
        return self.__b
    
    @property
    def A(self):
        """Getter for activated output"""
        return self.__A
