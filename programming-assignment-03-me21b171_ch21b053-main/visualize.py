import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

def visualize_smdp_q_values(agent, env, options, num_primitives=6):
    """
    Visualize the learned Q-values for SMDP Q-learning.
    """
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    fig.suptitle("SMDP Q learning with the provided options", fontsize=16)
    
    # Define colors for visualization
    cmap = plt.cm.RdBu_r
    
    # Map action indices to names
    primitive_actions = ['south', 'north', 'east', 'west', 'pickup', 'dropoff']
    option_names = ['go to R', 'go to G', 'go to Y', 'go to B']
    action_names = primitive_actions + option_names
    
    # For each destination and passenger location
    for dest_idx in range(4):  # R, G, Y, B destinations
        for pass_idx in [0, 4]:  # 0: at R, 4: in Taxi
            # Determine subplot position
            row = dest_idx
            col = 0 if pass_idx == 0 else 2  # First two columns for passenger at R, last two for in Taxi
            
            # Get Q-values for this state configuration
            q_values = np.zeros((5, 5))
            policy = np.zeros((5, 5), dtype=int)
            text_policy = np.empty((5, 5), dtype=object)
            
            # Extract Q-values for each position
            for row_idx in range(5):
                for col_idx in range(5):
                    # Convert position to state index
                    state = env.unwrapped.encode(row_idx, col_idx, pass_idx, dest_idx)
                    
                    # Get available actions/options
                    available = agent.get_available(state)
                    
                    # Get Q-values for this state
                    q_vals = agent.Q[state, available]
                    best_action_idx = available[np.argmax(q_vals)]
                    
                    # Store maximum Q-value and best action
                    q_values[row_idx, col_idx] = np.max(q_vals)
                    policy[row_idx, col_idx] = best_action_idx
                    
                    # Convert action index to name
                    if best_action_idx < num_primitives:
                        text_policy[row_idx, col_idx] = primitive_actions[best_action_idx]
                    else:
                        opt_idx = best_action_idx - num_primitives
                        text_policy[row_idx, col_idx] = option_names[opt_idx]
            
            # Plot Q-values
            ax1 = axes[row, col]
            im1 = ax1.imshow(q_values, cmap=cmap)
            ax1.set_title(f"passenger location: {'R' if pass_idx==0 else 'in Taxi'}, destination:{'RGBY'[dest_idx]}")
            ax1.set_xlabel("learned policy")
            
            # Add text annotations for actions
            for i in range(5):
                for j in range(5):
                    ax1.text(j, i, text_policy[i, j], ha="center", va="center", 
                             color="black", fontsize=8)
            
            # Plot state values
            ax2 = axes[row, col+1]
            im2 = ax2.imshow(q_values, cmap=cmap)
            ax2.set_title(f"passenger location: {'R' if pass_idx==0 else 'in Taxi'}, destination:{'RGBY'[dest_idx]}")
            ax2.set_ylabel("state values")
            
            # Add text annotations for values
            for i in range(5):
                for j in range(5):
                    ax2.text(j, i, f"{q_values[i, j]:.1f}", ha="center", va="center", 
                             color="black", fontsize=8)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("smdp_q_learning.png", dpi=300)
    plt.show()

def visualize_option_policies(options, env):
    """
    Visualize the policies for each option.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle("Policy for the provided options", fontsize=16)
    
    # Define colors for different actions
    colors = {
        'north': plt.cm.Purples(0.7),
        'south': plt.cm.Blues(0.7),
        'east': plt.cm.Oranges(0.7),
        'west': plt.cm.Reds(0.7)
    }
    
    option_names = ['go_to_R', 'go_to_G', 'go_to_Y', 'go_to_B']
    action_names = ['south', 'north', 'east', 'west']
    
    # For each option
    for i, option in enumerate(options):
        row, col = i // 2, i % 2
        ax = axes[row, col]
        
        # Create a grid to visualize the policy
        policy_grid = np.empty((5, 5), dtype=object)
        color_grid = np.zeros((5, 5, 4))  # RGBA values
        
        # For each position in the grid
        for row_idx in range(5):
            for col_idx in range(5):
                # Create a dummy state
                state = env.unwrapped.encode(row_idx, col_idx, 0, 0)  # Passenger and destination don't matter for option policy
                
                # Get the action from the option policy
                action = option.get_action(state)
                policy_grid[row_idx, col_idx] = action_names[action]
                
                # Assign color based on action
                color_grid[row_idx, col_idx] = colors[action_names[action]]
        
        # Plot the policy
        ax.imshow(color_grid)
        ax.set_title(f"Policy for option {option_names[i]}")
        
        # Add text annotations
        for i in range(5):
            for j in range(5):
                ax.text(j, i, policy_grid[i, j], ha="center", va="center", color="black")
        
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.set_xticklabels(range(5))
        ax.set_yticklabels(range(5))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("option_policies.png", dpi=300)
    plt.show()

def visualize_full_smdp_q_values(agent, env, options, num_primitives=6):
    """
    Create a comprehensive visualization of all passenger-destination combinations
    """
    # Create a large figure with 4 rows (destinations) and 5 columns (passenger locations)
    fig, axes = plt.subplots(4, 5, figsize=(25, 20))
    fig.suptitle("SMDP Q learning with the provided options - Complete visualization", fontsize=16)
    
    # Define locations
    locations = ['R', 'G', 'Y', 'B', 'in Taxi']
    
    # For each destination and passenger location
    for dest_idx in range(4):  # R, G, Y, B destinations
        for pass_idx in range(5):  # R, G, Y, B, in Taxi
            # Get Q-values for this state configuration
            q_values = np.zeros((5, 5))
            policy = np.zeros((5, 5), dtype=int)
            text_policy = np.empty((5, 5), dtype=object)
            
            # Extract Q-values for each position
            for row_idx in range(5):
                for col_idx in range(5):
                    # Convert position to state index
                    state = env.unwrapped.encode(row_idx, col_idx, pass_idx, dest_idx)
                    
                    # Get available actions/options
                    available = agent.get_available(state)
                    
                    # Get Q-values for this state
                    q_vals = agent.Q[state, available]
                    best_action_idx = available[np.argmax(q_vals)]
                    
                    # Store maximum Q-value and best action
                    q_values[row_idx, col_idx] = np.max(q_vals)
                    policy[row_idx, col_idx] = best_action_idx
                    
                    # Store action name
                    if best_action_idx < num_primitives:
                        text_policy[row_idx, col_idx] = ['S', 'N', 'E', 'W', 'P', 'D'][best_action_idx]
                    else:
                        opt_idx = best_action_idx - num_primitives
                        text_policy[row_idx, col_idx] = f"→{locations[opt_idx]}"
            
            # Plot Q-values with action overlay
            ax = axes[dest_idx, pass_idx]
            im = ax.imshow(q_values, cmap='RdBu_r')
            ax.set_title(f"pass:{locations[pass_idx]}, dest:{locations[dest_idx]}")
            
            # Add text annotations for policies
            for i in range(5):
                for j in range(5):
                    ax.text(j, i, text_policy[i, j], ha="center", va="center", 
                             color="black", fontsize=8)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("smdp_q_learning_full.png", dpi=300)
    plt.show()
