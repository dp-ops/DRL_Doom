#This repo is made for a project in Deep Learning and Multimedia Data Analysis in MCS Artificial Inteligence of Aistotle University of Thessaloniki.

# DRL Doom Project

This project implements Deep Reinforcement Learning (DRL) techniques to train agents to play Doom using the ViZDoom environment. The repository includes training scripts, utility functions, and custom gym environments for experimentation.

## Project Structure

- **DRL_Doom/train/**: Contains training scripts and configurations.
- **utils/**: Includes utility functions such as `DoomGym.py` for custom gym environments.
- **DRL_Doom/github/ViZDoom/**: Contains the ViZDoom environment files.
  - **scenarios/**: Includes predefined Doom scenarios (`basic.cfg`, `defend_the_center.cfg`, `deadly_corridor_s1.cfg`) for training and testing.

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd DRL_Doom
   ```

2. **Install Dependencies**:
   Ensure you have Python installed. Then, install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up ViZDoom**:
   - Download and install the ViZDoom engine from [ViZDoom GitHub](https://github.com/mwydmuch/ViZDoom).
   - Ensure the `scenarios/` folder contains the necessary `.cfg` files for the Doom scenarios.

## Usage
In the testing baseline.ipynb you can see how the agent and env is initialized and how to run the project.

### Custom Gym Environments
The `DoomGym.py` file provides custom OpenAI Gym environments for different Doom scenarios:
- `VizDoomGym`: For the `basic.cfg` scenario.
- `VizDoomGym_DC`: For the `defend_the_center.cfg` scenario.
- `VizDoomGym_DeadlyCorridor`: For the `deadly_corridor_s1.cfg` scenario.

Example usage:
```python
import gym
from utils.DoomGym import VizDoomGym

env = VizDoomGym(render=True, config="github/ViZDoom/scenarios/basic.cfg")
state = env.reset()
# Interact with the environment
action = env.action_space.sample()
state, reward, terminated, truncated, info = env.step(action)
```

## Reward Shaping
The `VizDoomGym_DeadlyCorridor` environment includes reward shaping based on:
- Movement reward.
- Damage taken and inflicted.
- Ammo usage.

This helps the agent learn more complex behaviors in challenging scenarios.

## Visualization
To visualize the logs, run the following on the log file.

   ```bash
   tensorboard --logdir=.
   ```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
