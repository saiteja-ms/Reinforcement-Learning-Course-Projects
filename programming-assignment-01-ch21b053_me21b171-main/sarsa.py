import numpy as np
from tqdm import tqdm
from policies import choose_action_softmax, choose_action_epsilon

def sarsa(env, Q, episodes, gamma=0.99, alpha=0.1, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05, print_freq=100, plot_heat=False, choose_action=choose_action_epsilon):
    """
    Generic Sarsa implementation.
    
    Parameters:
      env          : An environment that returns a discretized state as a tuple.
      Q            : Q-table (NumPy array) with shape (..., n_actions).
      episodes     : Total number of episodes to run.
      gamma        : Discount factor.
      alpha        : Learning rate.
      epsilon0     : Initial epsilon value (used only for epsilon-greedy).
      print_freq   : How often (in episodes) to print status.
      plot_heat    : (Unused) Option to plot a heatmap of Q-values.
      choose_action: Policy function for action selection.
                     For epsilon-greedy, this function expects an additional epsilon parameter.
    
    Returns:
      Tuple of arrays: (episode_rewards, steps_to_completion)
    """

    episode_rewards = np.zeros(episodes)
    steps_to_completion = np.zeros(episodes)


    for ep in tqdm(range(episodes)):
        tot_reward = 0
        steps = 0

        # Reset the environment and get the initial discretized state.
        state = env.reset()

        action = choose_action(Q, state, epsilon)

        done = False
        while not done:
            # Take a step in the environment.
            state_next, reward, done = env.step(action)

            # Select next action using the same policy.
            next_action = choose_action(Q, state_next, epsilon)
            
            # Sarsa update
            Q[tuple(state) + (action,)] += alpha * (reward + gamma * Q[tuple(state_next) + (next_action,)] - Q[tuple(state) + (action,)])

            tot_reward += reward
            steps += 1

            state, action = state_next, next_action

        episode_rewards[ep] = tot_reward
        steps_to_completion[ep] = steps

        if (ep + 1) % print_freq == 0:
            avg_reward = np.mean(episode_rewards[ep-print_freq+1:ep+1])
            avg_steps = np.mean(steps_to_completion[ep-print_freq+1:ep+1])
            print(f"Episode {ep+1}: Avg Reward: {avg_reward:.2f}, Avg Steps: {avg_steps:.2f}, Qmax: {Q.max():.2f}, Qmin: {Q.min():.2f}")

        # Decay epsilon after each episode
        epsilon = max(epsilon * epsilon_decay, epsilon_min)

    return episode_rewards, steps_to_completion


    

    
    