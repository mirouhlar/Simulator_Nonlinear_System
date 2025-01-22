# simulation_model.py

from abc import ABC, abstractmethod

class SimulationModel(ABC):
    """
    Abstract base class for a simulation model.
    Provides a common interface for all simulation models.
    """

    @abstractmethod
    def dynamics(self, t, state, control_input):
        """
        Compute the derivatives for the system.
        
        Parameters:
            t (float): Time variable.
            state (array): Current state vector.
            control_input (float or array): Control input to the system.
        
        Returns:
            derivatives (array): Array of state derivatives.
        """
        pass

    @abstractmethod
    def simulate(self, initial_state, duration, dt, control_func, method):
        """
        Simulate the model dynamics over a given time period.
        
        Parameters:
            initial_state (array): Initial state vector.
            duration (float): Duration of the simulation (seconds).
            dt (float): Time step for simulation.
            control_func (function): Function that provides control input based on time and state.
        
        Returns:
            (array, array): Arrays of time points and states.
        """
        pass

    @abstractmethod
    def linearize(self):
        """
        Linearize the dynamics around choosen point for LQR or other linear controllers.
        
        Returns:
            A, B (arrays): Linearized state-space matrices.
        """
        pass
