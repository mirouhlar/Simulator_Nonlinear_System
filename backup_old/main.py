import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple

class TransferFunction:
    def __init__(self, numerator: List[float], denominator: List[float], system_type: str = 'continuous', dt: float = 0.01):
        self.numerator = numerator
        self.denominator = denominator
        self.system_type = system_type
        self.dt = dt

    def bode_plot(self, start_freq: float, stop_freq: float, num_points: int):
        # Generate frequency points
        omega = np.logspace(np.log10(start_freq), np.log10(stop_freq), num_points)
        
        # Calculate the frequency response
        H = np.array([self._frequency_response(w) for w in omega])

        # Calculate magnitude and phase
        magnitude = np.abs(H)
        phase = np.angle(H, deg=True)

        # Convert magnitude to dB
        magnitude_db = 20 * np.log10(magnitude)
        
        # Plot the Bode plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        ax1.semilogx(omega, magnitude_db)
        ax1.set_title('Bode Plot')
        ax1.set_ylabel('Magnitude (dB)')
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

        ax2.semilogx(omega, phase)
        ax2.set_xlabel('Frequency (rad/s)')
        ax2.set_ylabel('Phase (degrees)')
        ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

        plt.tight_layout()
        plt.show()

    def _frequency_response(self, omega: float) -> complex:
        jw = 1j * omega
        numerator = sum([self.numerator[i] * (jw**(len(self.numerator) - i - 1)) for i in range(len(self.numerator))])
        denominator = sum([self.denominator[i] * (jw**(len(self.denominator) - i - 1)) for i in range(len(self.denominator))])
        return numerator / denominator

    def step_response(self, t_end: float) -> Tuple[List[float], List[float]]:
        if self.system_type == 'continuous':
            return self._continuous_step_response(t_end)
        elif self.system_type == 'discrete':
            return self._discrete_step_response(t_end)
        else:
            raise ValueError("system_type must be 'continuous' or 'discrete'")

    def _continuous_step_response(self, t_end: float) -> Tuple[List[float], List[float]]:
        # Initialize variables
        num_order = len(self.numerator)
        den_order = len(self.denominator)
        order = max(num_order, den_order)
        
        # State-space representation
        A = [[0 for _ in range(order-1)] for _ in range(order-1)]
        for i in range(order-2):
            A[i+1][i] = 1
        A[-1] = [-self.denominator[i] for i in range(1, den_order)]

        B = [0 for _ in range(order-1)]
        B[-1] = 1

        C = [self.numerator[i] - self.denominator[i] * self.numerator[0] / self.denominator[0] if i < num_order else - self.denominator[i] * self.numerator[0] / self.denominator[0] for i in range(1, den_order)]
        D = self.numerator[0] / self.denominator[0]

        # Simulation
        time = [i*self.dt for i in range(int(t_end/self.dt))]
        y = []
        x = [0 for _ in range(order-1)]
        u = 1  # step input

        for t in time:
            y.append(sum(C[i]*x[i] for i in range(order-1)) + D*u)
            dx = [sum(A[i][j]*x[j] for j in range(order-1)) + B[i]*u for i in range(order-1)]
            x = [x[i] + dx[i]*self.dt for i in range(order-1)]
        
        return time, y

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

# # Example usage
# # Continuous system: H(s) = (s + 1) / (s^2 + 2s + 1)
# numerator_cont = [1, 1]
# denominator_cont = [1, 2, 1]

# # Discrete system: H(z) = (z + 0.5) / (z^2 - 1.5z + 0.7)
# numerator_disc = [1, 0.5]
# denominator_disc = [1, -1.5, 0.7]

# # Create transfer function objects
# tf_cont = TransferFunction(numerator_cont, denominator_cont, system_type='continuous', dt=0.01)
# tf_disc = TransferFunction(numerator_disc, denominator_disc, system_type='discrete', dt=0.1)

# # Simulate step response
# t_end = 10
# time_cont, response_cont = tf_cont.step_response(t_end)
# time_disc, response_disc = tf_disc.step_response(t_end)

# # Plot the results
# plt.figure()
# plt.plot(time_cont, response_cont, label='Continuous System')
# plt.plot(time_disc, response_disc, label='Discrete System')
# plt.xlabel('Time')
# plt.ylabel('Response')
# plt.title('Step Response')
# plt.legend()
# plt.grid(True)
# plt.show()

# Example usage
# Transfer function: H(s) = (s + 1) / (s^2 + 2s + 1)
numerator = [1, 1]
denominator = [1, 2, 1]

# Create transfer function object
tf = TransferFunction(numerator, denominator)

# Generate Bode plot
start_freq = 0.1
stop_freq = 100
num_points = 1000
tf.bode_plot(start_freq, stop_freq, num_points)