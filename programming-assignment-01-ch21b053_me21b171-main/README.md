# This repository contains the implementation of SARSA and Q-Learning algorithms for the RL Assignment - 1.

## Team Members
- **Name:** Kush Shah, **Roll Number:** CH21B053
- **Name:** Sai Teja MS, **Roll Number:** ME21B171

## Environments:
- **CartPole-v1**
- **MountainCar-v0**
- **MiniGrid-Dynamic-Obstacles-5x5-v0(Bonus)**
  
## Running the Code
To run the experiments with the best hyperparameters, use the following commands:

```bash
# Run SARSA on CartPole-v1 with 10000 episodes
python run_experiment.py --env CartPole-v1 --algorithm sarsa --episodes 10000 --n_bins 25 --alpha 0.1 --epsilon_decay 0.995

# Run Q-Learning on CartPole-v1 with 10000 episodes
python run_experiment.py --env CartPole-v1 --algorithm qlearning --episodes 10000 --n_bins 20 --alpha 0.1 --tau_decay 0.9999

# Run SARSA on MountainCar-v0 with 15000 episodes
python run_experiment.py --env MountainCar-v0 --algorithm sarsa --episodes 15000 --n_bins 20 --alpha 0.05 --epsilon_decay 0.995

# Run Q-Learning on MountainCar-v0 with 15000 episodes
python run_experiment.py --env MountainCar-v0 --algorithm qlearning --episodes 10000 --n_bins 20 --alpha 0.1 --tau_decay 0.995
```

## The images for the sweeps have been added to the report.

## To run the Bonus task
Go to the **Bonus activity** Folder, and just run the following command:
```bash
python run_experiment.py
```
It will take care of itself. You can adjust the no of sweeps, bins,etc. as you wish to attain the optimum result.



