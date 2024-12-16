# MDP Project: Angry Birds Simulation

## Overview
This project implements a Markov Decision Process (MDP) to solve a problem inspired by the *Angry Birds* game. The goal is to design an agent that can navigate a stochastic grid-based environment, interact with elements such as pigs and eggs, and maximize its score using an optimal policy.

## Environment
The environment is a stochastic 8x8 grid with various elements:

- **Agent**: Starts at the top-left corner (0,0).
- **Eggs**: Located at the bottom-right corner (7,7). Reaching an egg grants positive rewards.
- **Pigs**: Positioned randomly in the grid. Destroying pigs grants positive rewards.
- **Obstacles**: Represented by rocks that block the agent's path.
- **Queens**: Randomly positioned. Colliding with a queen results in negative rewards.

The agent can move up, down, left, or right, but due to stochasticity, it may not always move in the intended direction. Each action has an associated reward or penalty, which is determined by proximity to pigs, eggs, or obstacles.

## Features
- **Dynamic Grid Environment**
- **Algorithms**:
  - **Value Iteration**: Computes optimal policies by iteratively refining value functions.
  - **Policy Iteration**: Optimizes decision-making policies for the agent.
- **Visualization**:
  - Heatmaps to display value functions (V*).
  - Plots tracking the convergence of the value iteration algorithm.
- **Logging**:
  - Logs rewards, policies, and value tables for debugging and analysis.


### How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/Yonesj/AngryAgent-MDP.git
   cd AngryAgent-MDP
   ```
2. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```
   
3. Run the main script:
   ```bash
   python src/main.py
   ```
4. Check the generated logs in the `logs` directory and visualizations in the `plot` directory.


## Files
- `main.py`: Entry point for running the simulation.
- `environment.py`: Represents the Angry Birds environment for a MDP.
- `policy.py`: Implements value iteration and policy iteration algorithms.
- `logger.py`: Logs the rewards, policies, and value tables for each iteration.
- `const.py`: Contains constants such as grid size, FPS and rewards.
- `util.py`: Contains utility functions and datastructures for the Angry Birds MDP.


## License
This project is licensed under the Apache License v2.0 License. See the `LICENSE` file for details.
