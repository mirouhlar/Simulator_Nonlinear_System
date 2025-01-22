# inverted_pendulum.py 
# planar inverted pendulum model

import numpy as np
from scipy.integrate import solve_ivp
from models.simulation_model import SimulationModel
import sympy as sp

class InvertedPendulum(SimulationModel):
    """
    Inverted pendulum on a cart simulation model.
    Inherits from the SimulationModel base class.
    """

    def __init__(self, mass_cart=0.3, mass_pendulum=0.275, length=0.5, d_c=0.3, d_p=0.01148, gravity=9.81):
        """
        Initialize the inverted pendulum parameters.
        """
        self.mc = mass_cart
        self.mp = mass_pendulum
        self.l = length
        self.g = gravity
        self.dc = d_c
        self.dp = d_p
        self.C = np.eye(4)  # Measurement matrix
        self.nx = 4 # number of states
        self.nu = 1 # number of inputs

    def dynamics(self, t, state, F):
        """
        Compute the derivatives for the equations for the inverted pendulum system.
        """
        x, theta, x_dot, theta_dot = state
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        I = (1 / 12) * self.mp * self.l**2

        com_div = 4 * I * (self.mc + self.mp) + self.l**2 * self.mp * (self.mc + self.mp - self.mp * cos_theta**2)
        numerator_1 = (
            8 * F * I 
            - 8 * I * self.dc * x_dot 
            + 2 * F * self.l**2 * self.mp
            - 2 * self.dc * self.l**2 * self.mp * x_dot 
            + self.l**3 * self.mp**2 * theta_dot**2 * sin_theta
            - 2 * self.g * self.l**2 * self.mp**2 * cos_theta * sin_theta
            + 4 * I * self.l * self.mp * theta_dot**2 * sin_theta
            + 4 * self.dp * self.l * self.mp * theta_dot * cos_theta
        )

        numerator_2 = - (
            4 * self.dp * self.mc * theta_dot 
            + 4 * self.dp * self.mp * theta_dot
            + 2 * F * self.l * self.mp * cos_theta
            - 2 * self.g * self.l * self.mp**2 * sin_theta
            + self.l**2 * self.mp**2 * theta_dot**2 * cos_theta * sin_theta
            - 2 * self.dc * self.l * self.mp * x_dot * cos_theta
            - 2 * self.g * self.l * self.mp * self.mc * sin_theta
        )

        x_ddot = numerator_1 / (2 * com_div)
        theta_ddot = numerator_2 / com_div

        return [x_dot, theta_dot, x_ddot, theta_ddot]


    def linearize(self):
        """
        Linearize the dynamics around the upright position for LQR or other linear controllers.
        """
        g, mc, mp, delta_x, delta_theta, l = self.g, self.mc, self.mp, self.dc, self.dp, self.l
        A = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, -(3 * g * mp) / (4 * mc + mp), 
             -(4 * delta_x) / (4 * mc + mp), 
             (6 * delta_theta) / (l * (4 * mc + mp))],
            [0, 
             (6 * g) * (mc + mp) / (l * (4 * mc + mp)), 
             (6 * delta_x) / (l * (4 * mc + mp)), 
             -(12 * delta_theta * (mc + mp)) / (l**2 * mp * (4 * mc + mp))]
        ])
        B = np.array([
            [0],
            [0],
            [4 / (4 * mc + mp)],
            [-6 / (l * (4 * mc + mp))]
        ])
        return A.round(3), B.round(3)
    
        
    # Override the __str__ method to customize the print output
    def __str__(self):
        return (f"Parameters of Inverted Pendulum:\n\n"
                f"    Mass of the cart (kg) -> {self.mc}\n"
                f"    Mass of the pendulum (kg) -> {self.mp}\n"
                f"    Length of the pendulum (m) -> {self.l}\n"
                f"    Gravitational acceleration (m/s^2) -> {self.g}\n"
                f"    Friction coefficient of the cart -> {self.dc}\n"
                f"    Damping coefficient of the pendulum -> {self.dp}")   
    

    def print_symbolic_model(self, linear = False):
         
        # Define the state variables and input
        t = sp.symbols('t')
        theta = sp.Function('theta')(t)
        theta_dot = sp.diff(theta, t)
        x = sp.Function('x')(t)
        x_dot = sp.diff(x, t)
        Fx = sp.Function('Fx')(t)  # Control input

        # Define constants
        mc, mp, l, I, g = sp.symbols('mc mp l I g')
        delta_x, delta_theta = sp.symbols('delta_x delta_theta')

        # Define the matrices with the original expressions
        S = sp.Matrix([
            [mc + mp, sp.Rational(1,2) * mp * l * sp.cos(theta)],
            [sp.Rational(1,2) * mp * l * sp.cos(theta), I + sp.Rational(1,4) * mp * l**2]])

        L = sp.Matrix([
            [delta_x, -sp.Rational(1,2) * mp * l * theta_dot * sp.sin(theta)],
            [0, delta_theta]])

        M = sp.Matrix([
            [0],
            [-sp.Rational(1,2) * mp * g * l * sp.sin(theta)]])

        N = sp.Matrix([
            [1],
            [0]])

        I1 = sp.Rational(1,12)*mp*l**2

        # Calculate S^(-1)
        S_inv = S.inv()

        # Define q_dot vector
        q_dot = sp.Matrix([x_dot, theta_dot])

        # Define the main expression S^(-1) * (N * Fx - L * q_dot - M)
        main_expr = S_inv * (N * Fx - L * q_dot - M)        

        state_vars = sp.Matrix([x, theta, x_dot, theta_dot])
        xdot = sp.Matrix.vstack(q_dot, main_expr)

        # Define the equilibrium point (zero point)
        equilibrium_point = {x: 0, theta: 0, x_dot: 0, theta_dot: 0, Fx: 0}

        # Linearize by computing Jacobians at the equilibrium point
        A = xdot.jacobian(state_vars).subs(equilibrium_point).subs({I: I1})
        B = xdot.jacobian([Fx]).subs(equilibrium_point).subs({I: I1})


        if linear:
            print("Linearized model of pendulum: \n")
            # Display the resulting A and B matrices
            print("A matrix: ")
            sp.pprint(sp.simplify(A)) 
            print("\nB matrix: ")
            sp.pprint(sp.simplify(B))
        else:    
            print("Nonlinear model of Inverted Pendulum: \n")
            print("S matrix: ")
            sp.pprint(sp.simplify(S)) 
            print("\nL matrix: ")
            sp.pprint(sp.simplify(L))
            print("\nM matrix: ")
            sp.pprint(sp.simplify(M)) 
            print("\nN matrix: ")
            sp.pprint(sp.simplify(N))

        print("\nState vector: ")
        sp.pprint(state_vars)
        print("\nInput variable: ")
        sp.pprint(Fx)

 
    def simulate(self, initial_state, duration, dt, control_func, method="RK45"):
        """
        Simulate the pendulum's dynamics over time using iterative stepping.

        Parameters:
            initial_state (array): Initial state [x, x_dot, theta, theta_dot].
            duration (float): Duration of the simulation (s).
            dt (float): Time step for output (s).
            control_func (function): Function that takes (t, state) and returns a control force (N).
            method (string): Method for integration

        Returns:
            time_points (array): Array of time points.
            states (array): Array of state vectors at each time point.
        """
        
        def wrapped_dynamics(t, state):
            force = control_func(t, state)
            return self.dynamics(t, state, force)
            # return A@state + B@np.array([force])

        time_points = np.arange(0, duration, dt)
        states = [initial_state]
        current_state = initial_state

        # Step through each time increment using solve_ivp
        for t in time_points[:-1]:
            # Integrate dynamics from t to t + dt
            sol = solve_ivp(wrapped_dynamics, [t, t + dt], current_state, method=method)
            
            # Update current_state to the end of this time step
            current_state = sol.y[:, -1]
            states.append(current_state)

        # Convert results to a numpy array for easier handling
        states = np.array(states)
        return time_points, states
    