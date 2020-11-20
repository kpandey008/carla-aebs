import matplotlib.pyplot as plt
import numpy as np
import os


def plot_metrics(
    comp_dist, gt_dist, braking_action, p_values, fontsize=10,
    figsize=(8, 8), save=False, save_path=None, caption='fig'
):
    fontdict = {
        'fontsize': fontsize
    }
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize)
    fig.tight_layout(pad=3.0)

    # Plot distances
    axes[0, 0].plot(comp_dist)
    axes[0, 0].plot(gt_dist)
    axes[0, 0].set_xlabel('Simulation steps')
    axes[0, 0].set_ylabel('Distance(m)')
    axes[0, 0].set_title('Distance Comparison (Computed vs GT)', fontdict=fontdict)

    axes[0, 1].plot(np.abs(comp_dist - gt_dist))
    axes[0, 1].set_xlabel('Simulation steps')
    axes[0, 1].set_ylabel('Distance(m)')
    axes[0, 1].set_title('Absolute distance Error', fontdict=fontdict)

    # Plot the braking action
    axes[1, 0].plot(braking_action)
    axes[1, 0].set_xlabel('Simulation steps')
    axes[1, 0].set_ylabel('Brake intensity')
    axes[1, 0].set_title('Braking action policy', fontdict=fontdict)

    # Plot the p-values
    axes[1, 1].set_yticks(np.linspace(0, 1, num=5))
    axes[1, 1].set_autoscaley_on(False)
    axes[1, 1].scatter(np.arange(p_values.shape[-1]), p_values)
    axes[1, 1].set_xlabel('Simulation steps')
    axes[1, 1].set_ylabel('P-values')
    axes[1, 1].set_title('P-values', fontdict=fontdict)

    plt.show()

    if save:
        if not save_path:
            raise ValueError('Save path must be provided when saving the figure')
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f'{caption}.png'), bbox_inches='tight')
