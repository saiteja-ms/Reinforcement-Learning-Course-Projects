# This repository contains the implementation of both types of DQN_Duelling and Monte Carlo Reinforce algorithms for the RL Assignment - 2.

## Team Members
- **Name:** Kush Shah, **Roll Number:** CH21B053
- **Name:** Sai Teja MS, **Roll Number:** ME21B171

## Environments:
- **CartPole-v1**
- **Acrobot-v1**
  
## Code File Structure
project/

├── experiments/

│   ├── run_dqn.py        # DQN experiment runner

│   └── run_reinforce.py  # REINFORCE experiment runner

├── models/

│   ├── dueling_dqn.py    # Dueling-DQN implementation

│   └── reinforce.py      # REINFORCE implementation

├── utils/

│   ├── environment.py    # Environment setup

│   └── plotting.py       # Plotting utilities

├── dueling_dqn_wandb.py  # Hyperparameter tuning for Dueling-DQN

├── reinforce_wandb.py    # Hyperparameter tuning for REINFORCE

└── main.py               # Main execution script

## Running the Code
To run the experiments to compare the two types for each algorithm, the following commands can be used:

```bash
# Run Dueling-DQN on CartPole
python main.py --algorithm dqn --env CartPole-v1

# Run Dueling-DQN on Acrobot
python main.py --algorithm dqn --env Acrobot-v1

# Run REINFORCE on CartPole
python main.py --algorithm reinforce --env CartPole-v1

# Run REINFORCE on Acrobot
python main.py --algorithm reinforce --env Acrobot-v1
```

The number of episodes and number of seeds can also be changed through the command line using --episodes and --seeds respectively.

## Hyperparameter Tuning
The code for hyperparameter tuning is in separate files named dueling_dqn_wandb.py and reinforce_wandb.py. They can be run as follows:

```bash
python dueling_dqn_wandb.py

python reinforce_wandb.py
```

Running python dueling_dqn_wandb.py will create 2 separate wandb projects for the enironments with 2 sweeps each for the 2 types.

Runnning python reinforce_wandb.py will also give similar results

