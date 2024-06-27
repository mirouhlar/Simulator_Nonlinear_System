import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple


class TransferFunction:

    def __init__(self, numerator: List[float], denominator: List[float], systemType: str = 'c', dt: float = 0.01):
        """
        Initializes the TransferFunction object.

        Parameters:
        numerator (list): The numerator coefficients of the system's transfer function.
        denominator (list): The denominator coefficients of the system's transfer function.
        systemType (str, optional): The type of the system ('c' for continuous or 'd' for discrete). Defaults to 'c'.
        dt (float, optional): The time step for discrete systems. Defaults to 0.01.


        """
        self.numerator = numerator
        self.denominator = denominator
        self.systemType = systemType
        self.dt = dt

    def stepResponse(self, t_end: float) -> Tuple[List[float], List[float]]:
       if self.systemType == 'c' or self.systemType == 'continuous':
           return self._continuous_step_response(t_end)
       elif self.systemType == 'd' or self.systemType == 'discrete':
            return self._discrete_step_response(t_end)
       else:
           raise ValueError("system_type must be 'continuous'/'c' or 'discrete'/'d'")
       
    
    def _continuous_step_response(self, t_end: float) -> Tuple[List[float], List[float]]:
        pass