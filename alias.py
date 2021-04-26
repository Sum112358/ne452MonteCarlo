#Implementation is adapted from Devroye, Non-Uniform Random Variate Generatation available at http://luc.devroye.org/rnbookindex.html 

from typing import List
import math

import numpy as np

from numpy.core.defchararray import index

np.random.default_rng(0) # seed the RNG function

class aliasMethod:
    def __init__(self, distribution: np.ndarray) -> None:
        self.distribution: np.ndarray = distribution
        self.k: int = len(distribution)
        self.q: np.ndarray = np.zeros(self.k)
        self.J: np.ndarray = np.zeros(self.k, dtype=float)
        self.S: List[float] = []
        self.L: List[float]  = []
        self.buildTable()
    
    def buildTable(self) -> None:
        for index, prob in enumerate(self.distribution):
            self.q[index] = self.k*prob
            if self.q[index] < 1.0:
                self.S.append(index)
            else:
                self.L.append(index)
        while (not self.S and not self.L): 
            s = self.S.pop()
            l = self.L.pop()
            self.J[s] = l
            self.q[l] = (self.q[l] + self.q[s]) - 1 # this way to minimize rounding error
            if self.q[l] < 1.0:
                self.S.append(l)
            else:
                self.L.append(l)

    def draw(self) -> float:
        index: int = int(math.floor(np.random.rand()*self.k)) # get index for a random draw from the generated uniform distribution
        
        if np.random.rand() < self.q[index]:
            return index
        else:
            return self.J[index] 
