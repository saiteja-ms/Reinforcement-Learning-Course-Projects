import matplotlib.pyplot as plt

def plot_learning_curves(*args, labels=None):
    plt.figure(figsize=(10,6))
    for i, rewards in enumerate(args):
        plt.plot(rewards, label=labels[i] if labels else f'Agent {i+1}')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Learning Curves')
    plt.legend()
    plt.show()
