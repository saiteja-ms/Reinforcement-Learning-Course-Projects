# Hierarchical Reinforcement Learning with Options (RL Assignment - 3)
## Introduction

This repository implements hierarchical reinforcement learning algorithms with options framework in the Taxi-v3 environment. We explore two algorithms-SMDP Q-learning and Intra-option Q-learning-with two different option sets to compare their learning efficiency and performance(Intra-Options Q-Learning is implemented as a bonus task). Also, report has also been added in the repository.


## Team Members
- **Name:** Kush Shah, **Roll Number:** CH21B053  
- **Name:** Sai Teja MS, **Roll Number:** ME21B171  

## Contributions (Equal from both members):
1. Kush Shah: Worked on SMDP and Q-Learning Visualizations, and its corresponding parts of report and readme.
2. M S Sai Teja: Worked on SMDP and  Intra Options Q-Learning implementation, readme and corresponding parts of Intra Options in report.
   
## Environment:
- **Taxi-v3**  
  - State space: 500 discrete states  
  - Actions: 6 primitive + 4 navigation options  
  - Reward: -1/step, +20 for success, -10 for illegal actions  

## Algorithms

### SMDP Q-Learning

SMDP Q-learning treats options as temporally extended actions. When an option is selected in state s:

- The agent follows the option's policy until termination
- Accumulates discounted rewards during option execution
- Updates Q(s,o) using the cumulative reward and next state value:

```
Q(s,o) ← Q(s,o) + α[R + γᵏ·max Q(s',o') - Q(s,o)]
```


Where k is the number of steps the option took to complete.

### Intra-Option Q-Learning

Intra-option Q-learning updates option values after every primitive action:

- For every option that would have selected the executed action in the current state
- Updates its value considering whether the option would terminate in the next state
- Allows faster learning by using more experiences to update option values:

```
Q(s,o) ← Q(s,o) + α[r + γ·((1-β)·Q(s',o) + β·max Q(s',a')) - Q(s,o)]
```

Where β=1 if option o terminates in s', otherwise β=0.

## Option Sets

### Original Option Set

Four navigation options that move the taxi to the landmark locations (R, G, Y, B):

- Each option uses a greedy policy (vertical then horizontal movement)
- Options terminate when the taxi reaches the target location
- Options are only available when not already at their target


### Passenger-Focused Option Set

A task-oriented option that dynamically selects targets based on passenger status:

- Navigates to passenger location for pickup when passenger not in taxi
- Navigates to destination when passenger is in taxi
- Automatically handles pickup/dropoff actions when at appropriate locations

## Code File Structure
```
project/
├── agents/
│   ├── smdp_q_learner.py      # SMDP Q-learning implementation
│   └── intra_option_learner.py # Intra-option Q-learning
├── options/
│   ├── taxi_options.py        # Landmark navigation options
│   └── passenger_options.py   # Passenger-focused options
├── config.py                  # Hyperparameters &amp; option sets
├── run_smdp.py                # SMDP experiment runner
├── run_intra.py               # Intra-option experiment runner
├── visualize.py               # Policy/Q-value visualization
├── utils.py                   # Utility functions
└── Intra_Option_Q_Learning(Bonus)  # Bonus implementation
└── RL_Assignment_3            # Assignment Report
```

## Running the Code
To run SMDP with the original option set and best hyperparameters use:

```bash
python run_smdp.py --option-set original --alpha 0.1 --gamma 0.9 --epsilon 0.01 --episodes 15000
```

Or simply running the following will also work since the best hyperparameters have been already set in config.py

```bash
python run_smdp.py --option-set original
``` 

To run SMDP with the passanger-focused option set and best hyperparameters use:

```
python run_smdp.py --option-set passenger_focused --alpha 0.1 --gamma 0.9 --epsilon 0.01 --episodes 15000
```


Or simply running the following will also work since the best hyperparameters have been already set in config.py

```
python run_smdp.py --option-set passenger_focused
```

### Running the code for bonus task
Go to the directory "Intra_Option_Q_Learning (Bonus):
1. Go into "Original Set Option" directory:
```
python run_experiments.py
```
(It will run both Intra-Options and SMDP algorithms and generate their plots for the Original Set Option), and if required you can uncomment the code for comparison plot generation).

2. Go into "Passenger Set Option" directory:
```
python run_experiments.py
```
(It will run both Intra-Options and SMDP algorithms and generate their plots for the Passenger Set Option, and if required you can uncomment the code for comparison plot generation).

## Key Results

### SMDP Q-Learning

- **Original Option Set**: Converges around episode 2000, stabilizing with rewards around 8.
- **Passenger-Focused Option Set**: Faster convergence with higher stability.

### Intra-Option Q-Learning

- **Original Option Set**: Shows rapid early learning, with agents recovering from initial -2500 rewards to near-optimal performance within 400 episodes
- **Passenger-Focused Option Set**: Less extreme initial performance (starting around -600 rewards) with gradual improvement

