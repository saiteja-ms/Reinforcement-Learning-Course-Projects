import argparse
import gymnasium as gym
import matplotlib.pyplot as plt
from config import HYPERPARAMS, EPISODES, OPTION_SETS
from options.taxi_options import TaxiOption
from options.passenger_options import PassengerOption
from agents.smdp_q_learner import SMDPQLearner
from visualize import visualize_smdp_q_values, visualize_option_policies, visualize_full_smdp_q_values

def parse_args():
    parser = argparse.ArgumentParser(description='Run SMDP Q-Learning with configurable parameters')
    parser.add_argument('--option-set', choices=OPTION_SETS.keys(),
                      default='original', help='Option set to use (default: original)')
    parser.add_argument('--alpha', type=float, default=HYPERPARAMS['alpha'],
                      help=f'Learning rate (default: {HYPERPARAMS["alpha"]})')
    parser.add_argument('--gamma', type=float, default=HYPERPARAMS['gamma'],
                      help=f'Discount factor (default: {HYPERPARAMS["gamma"]})')
    parser.add_argument('--epsilon', type=float, default=HYPERPARAMS['epsilon'],
                      help=f'Exploration rate (default: {HYPERPARAMS["epsilon"]})')
    parser.add_argument('--episodes', type=int, default=EPISODES,
                      help=f'Training episodes (default: {EPISODES})')
    return parser.parse_args()

def run(args):
    env = gym.make('Taxi-v3')
    
    # Initialize options
    if args.option_set == 'original':
        options = [TaxiOption(pos, env) for pos in OPTION_SETS[args.option_set]]
    elif args.option_set == 'passenger_focused':
        options = [PassengerOption(env)]
    
    # Initialize agent with command-line parameters
    agent = SMDPQLearner(
        env=env,
        options=options,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon
    )
    
    # Train with specified episodes
    rewards = agent.train(args.episodes)

    # Visualization
    print(f"Visualizing results for {args.option_set} options")
    visualize_smdp_q_values(agent, env, options)
    visualize_option_policies(options, env)
    visualize_full_smdp_q_values(agent, env, options)
    
    # Plot results with parameter info
    plt.plot(rewards, label=f'SMDP ({args.option_set})')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(f'α={args.alpha}, γ={args.gamma}, ε={args.epsilon}, Episodes={args.episodes}')
    plt.legend()
    plt.savefig(f'smdp_{args.option_set}.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    args = parse_args()
    run(args)
