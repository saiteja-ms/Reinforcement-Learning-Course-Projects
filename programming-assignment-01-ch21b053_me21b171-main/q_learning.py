import numpy as np
from tqdm import tqdm
from policies import choose_action_softmax, choose_action_epsilon

def q_learning(env, Q, episodes, gamma=0.99, alpha=0.1, 
               tau=1.0, tau_decay=0.995, tau_min=0.1, 
               print_freq=100, plot_heat=False, choose_action=choose_action_softmax):
    """
    Generic Q-learning implementation.
    
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


    for ep in tqdm(range(episodes), desc="Q-learning"):
        tot_reward = 0
        steps = 0

        # Reset the environment and get the initial discretized state.
        state = env.reset()

        action = choose_action(Q, state, tau)

        done = False
        while not done:
            # Take a step in the environment.
            state_next, reward, done = env.step(action)
            
            # Q-learning update
            best_next_value = np.max(Q[tuple(state_next)])
            Q[tuple(state) + (action,)] += alpha * (reward + gamma * best_next_value - Q[tuple(state) + (action,)])

            tot_reward += reward
            steps += 1

            # Update state and action for next iteration.
            state = state_next
            action = choose_action(Q, state, tau)

        episode_rewards[ep] = tot_reward
        steps_to_completion[ep] = steps

        if (ep + 1) % print_freq == 0:
            avg_reward = np.mean(episode_rewards[ep-print_freq+1:ep+1])
            avg_steps = np.mean(steps_to_completion[ep-print_freq+1:ep+1])
            print(f"Episode {ep+1}: Avg Reward: {avg_reward:.2f}, Avg Steps: {avg_steps:.2f}, Qmax: {Q.max():.2f}, Qmin: {Q.min():.2f}")

        tau = max(tau * tau_decay, tau_min)

    return episode_rewards, steps_to_completion


    

    
    