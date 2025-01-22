from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import scipy

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

    def step_response(self, t_end: float) -> Tuple[List[float], List[float]]:
        if self.systemType == 'c':
            return self._continuous_step_response(t_end)
        elif self.systemType == 'd':
            return self._discrete_step_response(t_end)
        else:
            raise ValueError("systemType must be 'c' for continuous or 'd' for discrete")

    def _continuous_step_response(self, t_end: float) -> Tuple[List[float], List[float]]:
        # Define time vector
        t = np.arange(0, t_end, self.dt)
        
        # Define the input (step function)
        u = np.ones_like(t)
        
        # Calculate the impulse response using the inverse Laplace transform approximation
        impulse_response = np.zeros_like(t)
        for i in range(len(t)):
            s = 1j * t[i]
            H_s = (np.polyval(self.numerator, s) / np.polyval(self.denominator, s)).real
            impulse_response[i] = H_s
        
        # Calculate the step response as the convolution of the impulse response with the step input
        step_response = np.convolve(impulse_response, u)[:len(t)] * self.dt

        return t.tolist(), step_response.tolist()

    def _discrete_step_response(self, t_end: float) -> Tuple[List[float], List[float]]:
        # Initialize variables
        num_order = len(self.numerator)
        den_order = len(self.denominator)
        order = max(num_order, den_order)
        
        # Initialize output and state vectors
        time = [i*self.dt for i in range(int(t_end/self.dt))]
        y = [0 for _ in range(len(time))]
        x = [0 for _ in range(order)]

        u = 1  # step input

        # Simulation using difference equations
        for n in range(1, len(time)):
            x_new = sum(self.numerator[i] * (u if n-i >= 0 else 0) for i in range(num_order))
            x_new -= sum(self.denominator[i] * (y[n-i] if n-i >= 0 else 0) for i in range(1, den_order))
            x_new /= self.denominator[0]
            y[n] = x_new
        
        return time, y

    def to_discrete(self, method='bilinear') -> 'TransferFunction':
        if method == 'bilinear':
            return self._bilinear_transformation()
        elif method == 'zoh':
            return self._zoh_transformation()
        else:
            raise ValueError("Method must be 'bilinear' or 'zoh'")

    def _bilinear_transformation(self) -> 'TransferFunction':
        # Bilinear transformation (Tustin's method)
        num_order = len(self.numerator)
        den_order = len(self.denominator)
        max_order = max(num_order, den_order)
        
        # Initialize discrete coefficients
        T = self.dt
        
        # Transform coefficients
        a_d = np.poly1d([0])
        b_d = np.poly1d([0])
        for i in range(max_order):
            a_coeff = self.denominator[i] if i < len(self.denominator) else 0
            b_coeff = self.numerator[i] if i < len(self.numerator) else 0
            a_d = np.polyadd(a_d, np.poly1d([a_coeff]) * (np.poly1d([2, -2])**(max_order - i - 1)) * (T / 2)**i)
            b_d = np.polyadd(b_d, np.poly1d([b_coeff]) * (np.poly1d([2, -2])**(max_order - i - 1)) * (T / 2)**i)
        
        # Normalize coefficients
        a_d_coeffs = a_d.coeffs
        b_d_coeffs = b_d.coeffs
        a_d_coeffs = np.real(a_d_coeffs / a_d_coeffs[0])
        b_d_coeffs = np.real(b_d_coeffs / a_d_coeffs[0])
        
        return TransferFunction(b_d_coeffs.tolist(), a_d_coeffs.tolist(), systemType='d', dt=self.dt)

    def _zoh_transformation(self) -> 'TransferFunction':
        """
        Performs ZOH transformation from continuous-time to discrete-time transfer function.

        Returns:
        TransferFunction: Discrete-time transfer function object.
        """
        T = self.dt

        # Pad numerator to match the length of denominator
        num = np.pad(self.numerator, (0, len(self.denominator) - len(self.numerator)), 'constant')

        # Construct the A and B matrices for ZOH transformation
        A = np.zeros((len(self.denominator) - 1, len(self.denominator) - 1))
        B = np.zeros((len(self.denominator) - 1, 1))

        A[:, 0] = -np.array(self.denominator[1:]) / self.denominator[0]
        if len(A) > 1:
            A[1:, :-1] = np.eye(len(self.denominator) - 2)

        B[0, 0] = 1 / self.denominator[0]

        # Construct the discrete-time system using matrix exponential
        M = np.vstack((np.hstack((A, B)), np.zeros((1, len(A) + 1))))
        Md = scipy.linalg.expm(M * T)

        Ad = Md[:len(A), :len(A)]
        Bd = Md[:len(A), len(A):]

        # Compute the discrete-time numerator and denominator
        num_z = np.dot(num, Bd[:, 0]) + num[0] * np.dot(self.denominator[1:], Ad[:, -1])
        den_z = np.poly(Ad)

        return TransferFunction(num_z.tolist(), den_z.tolist(), dt=self.dt)

    
    
    def __str__(self) -> str:
        """
        Returns a string representation of the TransferFunction object.
        """
        def format_polynomial(coeffs):
            terms = []
            order = len(coeffs) - 1
            for i, coeff in enumerate(coeffs):
                power = order - i
                if power > 1:
                    terms.append(f"{coeff:.2f}s^{power}")
                elif power == 1:
                    terms.append(f"{coeff:.2f}s")
                else:
                    terms.append(f"{coeff:.2f}")
            return " + ".join(terms)

        num_str = format_polynomial(self.numerator)
        den_str = format_polynomial(self.denominator)
        
        if self.systemType == 'c':
            system_type_str = 'Continuous'
        else:
            system_type_str = 'Discrete'
        
        order = max(len(self.numerator), len(self.denominator)) - 1
        info_str = (
            f"System Type: {system_type_str}\n"
            f"Sampling Period (dt): {self.dt}\n"
            f"System Order: {order}\n"
            "Transfer Function:\n"
            f"       {num_str}\n"
            f"H(s) = {'-' * len(den_str)}\n"
            f"       {den_str}"
        )
        return info_str


# Example usage
# Continuous system: H(s) = 1 / (s^2 + s + 0.2)
numerator_cont = [0.1]
denominator_cont = [1, 1, 0.2]

# Create continuous transfer function object
tf_cont = TransferFunction(numerator_cont, denominator_cont, systemType='c', dt=2)

# Convert to discrete using ZOH
tf_disc_zoh = tf_cont.to_discrete(method='zoh')

# Simulate step response for both continuous and discrete systems
t_end = 10
time_cont, response_cont = tf_cont.step_response(t_end)
time_disc_zoh, response_disc_zoh = tf_disc_zoh.step_response(t_end)

# # Plot the results
# plt.figure()
# plt.plot(time_cont, response_cont, label='Continuous System')
# plt.plot(time_disc_zoh, response_disc_zoh, label='Discrete System (ZOH)')
# plt.xlabel('Time')
# plt.ylabel('Response')
# plt.title('Step Response')
# plt.legend()
# plt.grid(True)
# plt.show()
print(tf_cont)
print(tf_disc_zoh)

