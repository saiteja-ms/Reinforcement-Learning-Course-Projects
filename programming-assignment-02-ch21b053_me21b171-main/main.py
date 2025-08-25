import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


import argparse
import os
from experiments.run_dqn import run_dueling_dqn_experiments
from experiments.run_reinforce import run_reinforce_experiments

def main():
    parser = argparse.ArgumentParser(description='Run RL experiments for DA6400 assignment')
    parser.add_argument('--env', type=str, choices=['Acrobot-v1', 'CartPole-v1'], 
                        default='CartPole-v1', help='Environment name')
    parser.add_argument('--algorithm', type=str, choices=['dqn', 'reinforce', 'all'], 
                        default='all', help='Algorithm to run')
    parser.add_argument('--episodes', type=int, default=500, 
                        help='Number of episodes per experiment')
    parser.add_argument('--seeds', type=int, default=5, 
                        help='Number of random seeds')
    parser.add_argument('--save_dir', type=str, default='results', 
                        help='Directory to save results')
    parser.add_argument('--gamma', type=float, default=0.99, 
                        help='Discount factor (fixed at 0.99 for assignment)')
    
    args = parser.parse_args()
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Run experiments for both environments if specified
    environments = ['Acrobot-v1', 'CartPole-v1'] if args.env == 'all' else [args.env]
    
    for env_name in environments:
        # Run experiments
        if args.algorithm in ['dqn', 'all']:
            print(f"\n=== Running Dueling DQN experiments on {env_name} ===")
            run_dueling_dqn_experiments(
                env_name=env_name,
                num_seeds=args.seeds,
                num_episodes=args.episodes,
                gamma=args.gamma,
                save_dir=args.save_dir
            )
        
        if args.algorithm in ['reinforce', 'all']:
            print(f"\n=== Running REINFORCE experiments on {env_name} ===")
            run_reinforce_experiments(
                env_name=env_name,
                num_seeds=args.seeds,
                num_episodes=args.episodes,
                gamma=args.gamma,
                save_dir=args.save_dir
            )

if __name__ == "__main__":
    main()
