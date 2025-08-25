import numpy as np
from tqdm import tqdm
from policies import choose_action_softmax

def q_learning(env, Q, episodes, gamma=0.99, alpha=0.1, tau=1.0, tau_decay=0.995, tau_min=0.1, print_freq=100, choose_action=choose_action_softmax):
    """
    Q-learning implementation with softmax exploration.
    """
    episode_rewards = np.zeros(episodes)
    steps_to_completion = np.zeros(episodes)

    for ep in tqdm(range(episodes), desc="Q-learning"):
        tot_reward = 0
        steps = 0

        # Reset the environment and get the initial state
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            state = reset_result[0]  # Handle both (state, _) and state returns
        else:
            state = reset_result

        # Choose initial action using softmax policy
        action = choose_action(Q, state, tau)

        done = False
        while not done:
            # Take a step in the environment
            step_result = env.step(action)
            
            # Handle different return formats
            if len(step_result) == 3:  # state, reward, done
                next_state, reward, done = step_result
            elif len(step_result) == 4:  # state, reward, done, info
                next_state, reward, done, _ = step_result
            else:  # state, reward, terminated, truncated, info
                next_state, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            
            # Q-learning update (off-policy)
            if isinstance(next_state, (int, np.integer)):
                best_next_value = np.max(Q[next_state])
            else:
                best_next_value = np.max(Q[tuple(next_state)])
                
            if isinstance(state, (int, np.integer)):
                current_q = Q[state][action]
                Q[state][action] = current_q + alpha * (reward + gamma * best_next_value - current_q)
            else:
                state_idx = tuple(state)
                current_q = Q[state_idx][action]
                Q[state_idx][action] = current_q + alpha * (reward + gamma * best_next_value - current_q)

            tot_reward += reward
            steps += 1

            # Update state and action for next iteration
            state = next_state
            action = choose_action(Q, state, tau)

        episode_rewards[ep] = tot_reward
        steps_to_completion[ep] = steps

        if (ep + 1) % print_freq == 0:
            avg_reward = np.mean(episode_rewards[ep-print_freq+1:ep+1])
            avg_steps = np.mean(steps_to_completion[ep-print_freq+1:ep+1])
            print(f"Episode {ep+1}: Avg Reward: {avg_reward:.2f}, Avg Steps: {avg_steps:.2f}, Tau: {tau:.4f}")

        # Decay tau for softmax exploration
        tau = max(tau * tau_decay, tau_min)

    return episode_rewards, steps_to_completion
