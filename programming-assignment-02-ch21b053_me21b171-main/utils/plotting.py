import numpy as np
import matplotlib.pyplot as plt

def plot_results(all_returns, title, variant_names=None):
    """
    Plot the results of experiments.
    
    Args:
        all_returns: List of returns for each variant and seed
        title: Plot title
        variant_names: Names of variants for legend
    """
    plt.figure(figsize=(10, 6))

    if variant_names is None:
        variant_names = [f"Type {i+1}" for i in range(len(all_returns))]

    colors = ['red', 'blue']

    for i, variant_returns in enumerate(all_returns):
        # Calculate mean and standard deviation across seeds
        variant_returns = np.array(variant_returns)
        mean_returns = np.mean(variant_returns, axis=0)
        std_returns = np.std(variant_returns, axis=0)

        # Plot mean line
        episodes = np.arange(1, len(mean_returns) + 1)
        plt.plot(episodes, mean_returns, label=variant_names[i], color=colors[i])

        # Plot shaded area for standard deviation
        plt.fill_between(episodes, mean_returns - std_returns, mean_returns + std_returns, alpha=0.2, color=colors[i])

    plt.title(title)
    plt.xlabel("Episode Number")
    plt.ylabel("Episodic Return")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    return plt
