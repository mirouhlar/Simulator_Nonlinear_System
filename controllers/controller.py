from abc import ABC, abstractmethod


class Controller(ABC):
    """
    Abstract base class for a controller.
    """
    
    @abstractmethod
    def compute_control(self, state):
        """
        Compute the control input based on the current state.
        
        Parameters:
        - state (np.ndarray): Current state vector.
        
        Returns:
        - u (np.ndarray): Control input.
        """
        pass
