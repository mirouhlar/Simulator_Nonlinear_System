import numpy as np
from scipy.linalg import solve_continuous_are, solve_discrete_are
from controllers.controller import Controller


class LQRController(Controller):
    """
    Linear Quadratic Regulator (LQR) controller for both continuous and discrete systems.
    """
    def __init__(self, A, B, Q, R, mode='continuous'):
        """
        Initialize the LQR controller.
        
        Parameters:
        - A  (np.ndarray): State-transition matrix.
        - B  (np.ndarray): Control input matrix.
        - Q  (np.ndarray): State cost matrix.
        - R  (np.ndarray): Control cost matrix.
        - mode (str): Mode of the system ('continuous' or 'discrete').
        """
        # Validate input dimensions
        if A.shape[0] != A.shape[1]:
            raise ValueError("Matrix A must be square.")
        if B.shape[0] != A.shape[0]:
            raise ValueError("Matrix B must have the same number of rows as A.")
        if Q.shape != A.shape:
            raise ValueError("Matrix Q must have the same dimensions as A.")
        if R.shape[0] != R.shape[1] or R.shape[0] != B.shape[1]:
            raise ValueError("Matrix R must be square and match the number of columns of B.")
        if mode not in ['continuous', 'discrete']:
            raise ValueError("Mode must be 'continuous' or 'discrete'.")

        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.mode = mode

        # Compute the gain matrix K during initialization
        self.K = self._compute_gain_matrix()

    def _compute_gain_matrix(self):
        """
        Compute the LQR gain matrix.
        """
        if self.mode == 'continuous':
            # Solve the Continuous Algebraic Riccati Equation (CARE)
            P = solve_continuous_are(self.A, self.B, self.Q, self.R)
            K = np.linalg.inv(self.R) @ self.B.T @ P
        elif self.mode == 'discrete':
            # Solve the Discrete Algebraic Riccati Equation (DARE)
            P = solve_discrete_are(self.A, self.B, self.Q, self.R)
            K = np.linalg.inv(self.B.T @ P @ self.B + self.R) @ (self.B.T @ P @ self.A)
        return K

    def compute_control(self, t, state):
        """
        Compute the control input based on the current state.
        
        Parameters:
        - state (np.ndarray): Current state vector.
        
        Returns:
        - u (np.ndarray): Control input.
        """
        u = -self.K.flatten() @ state
        return u
