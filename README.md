# Simulation of Nonlinear Systems

This repository contains code for simulating and controlling an inverted pendulum on a cart, a classic problem in the field of nonlinear dynamic systems modeling and control. The mathematical model and simulations here are part of ongoing research in nonlinear dynamics. Will be updated with more control strategies, systems etc.

## Overview
The inverted pendulum, system we are currently focused on, is a well-known benchmark system used to study dynamics and control strategies. It consists of a pendulum mounted on a cart, where the objective is to stabilize the pendulum in its upright position by applying forces to the cart.

This repository includes:
- A simulation model of the inverted pendulum.
- A controller framework for designing and testing control strategies.
- A Jupyter notebook for running simulations and visualizing results.

## Project Structure
- **`inverted_pendulum_sim.ipynb`**: A Jupyter notebook to simulate and visualize the behavior of the inverted pendulum. It demonstrates how the model and controller interact and provides plots of system performance.

- **`controller.py`**: Contains the abstract base class for controllers. This allows for the implementation of custom control strategies by inheriting and implementing the `compute_control` method.

- **`inverted_pendulum.py`**: Implements the mathematical model of the inverted pendulum. It includes:
  - Parameters for the system (e.g., mass, length, gravity).
  - Differential equations for the system dynamics.
  - Integration methods for simulating the pendulum's motion.

## Key Features
- **Modular Design**: The code is structured to separate the model, controller, and simulation logic, making it easy to test different control algorithms.
- **Customizable Parameters**: The pendulum and cart parameters can be modified to test various configurations.
- **Visualization**: The Jupyter notebook includes tools for visualizing system behavior, such as state trajectories and control inputs.

## How to Use
1. Clone this repository:
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter notebook:
   ```bash
   jupyter notebook inverted_pendulum_sim.ipynb
   ```

4. Implement a custom controller by inheriting from the `Controller` class in `controller.py` and overriding the `compute_control` method.

5. Simulate the system with your controller and visualize the results in the notebook.

## Dependencies
- Python 3.8+
- NumPy
- SciPy
- Matplotlib
- SymPy
- Jupyter Notebook

## Research Context
This project is part of broader research in the modeling and control of nonlinear dynamic systems. The inverted pendulum serves as a case study for developing advanced control strategies.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contributions
Contributions are welcome! If you'd like to contribute, please open an issue or submit a pull request.

## Acknowledgments
Special thanks to the nonlinear dynamics research community for their inspiration and foundational work in this area.
