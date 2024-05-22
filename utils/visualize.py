import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyemma.plots import plot_free_energy

def visualize_dihedral(dihedral, label, save_path, experiment_idx):
    """
    Visualize the results of embedding.

    Args:
    - traj_RE (numpy.ndarray): The embedded trajectory.
    - label (numpy.ndarray): Labels for the trajectory data points.
    - save_path (str): Path to save the visualization.
    - experiment_idx (int): Index of the experiment.
    """
    scatter = np.hstack([dihedral, label])
    d_scatter = pd.DataFrame(scatter, columns=['phi', 'xi', 'label'])
    colors = ['b', 'g', 'r', 'c', 'orange', 'pink']
    plt.figure(figsize=(10, 7))
    for index in range(6):
        plt.subplot(2, 3, index + 1)
        phi = d_scatter.loc[d_scatter['label'] == index]['phi']
        xi = d_scatter.loc[d_scatter['label'] == index]['xi']
        plt.xlim(dihedral[:, 0].min(), dihedral[:, 0].max())
        plt.ylim(dihedral[:, 1].min(), dihedral[:, 1].max())
        plt.scatter(dihedral[:, 0], dihedral[:, 1], c='gray', s=3)
        plt.scatter(phi.to_numpy(), xi.to_numpy(), c=colors[index], s=3)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plt.savefig(os.path.join(save_path, f'dihedral_scatter_{experiment_idx}.png'))
    plt.close()

    plot_free_energy(dihedral[:, 0], dihedral[:, 1], vmax=7)
    # plt.plot(dihedral[::10, 0], dihedral[::10, 1], '.', color='gray', alpha =0.1)
    plt.savefig(os.path.join(save_path, 'dihedral_free_energy.png'))
    plt.close()

def visualize_embedding(traj_RE, label, save_path, experiment_idx, args):
    """
    Visualize the results of embedding.

    Args:
    - traj_RE (numpy.ndarray): The embedded trajectory.
    - label (numpy.ndarray): Labels for the trajectory data points.
    - save_path (str): Path to save the visualization.
    - experiment_idx (int): Index of the experiment.
    """
    re = traj_RE[:, :2]
    scatter = np.hstack([re, label])
    d_scatter = pd.DataFrame(scatter, columns=['phi', 'xi', 'label'])
    colors = ['b', 'g', 'r', 'c', 'orange', 'pink']
    plt.figure(figsize=(10, 7))
    for index in range(6):
        plt.subplot(2, 3, index + 1)
        phi = d_scatter.loc[d_scatter['label'] == index]['phi']
        xi = d_scatter.loc[d_scatter['label'] == index]['xi']
        plt.xlim(re[:, 0].min(), re[:, 0].max())
        plt.ylim(re[:, 1].min(), re[:, 1].max())
        plt.scatter(re[:, 0], re[:, 1], c='gray', s=3)
        plt.scatter(phi.to_numpy(), xi.to_numpy(), c=colors[index], s=3)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    output_pre = f'type_{args.embeding_type}_end{args.diffusion_t_end}_step{args.diffusion_num_steps}'
    plt.savefig(os.path.join(save_path, f'em_scatter_only_embedding_{output_pre}_{experiment_idx}.png'))
    plt.close()

    plot_free_energy(traj_RE[:, 0], traj_RE[:, 1], vmax=7)
    # plt.plot(traj_RE[::10, 0], traj_RE[::10, 1], '.', color='gray', alpha =0.1)
    plt.savefig(os.path.join(save_path, f'embedding_free_energy_{output_pre}_{experiment_idx}.png'))
    plt.close()

def visualize_potential(X1, X2, tmp_traj_V, force, save_path, experiment_idx):
    """
    Visualize the potential derived from the model.

    Args:
    - X1, X2 (numpy.ndarray): Meshgrid arrays.
    - tmp_traj_V (numpy.ndarray): Potential values for the meshgrid.
    - save_path (str): Path to save the visualization.
    - experiment_idx (int): Index of the experiment.
    """
    plt.figure(figsize=(7, 6))
    plt.contourf(X1, X2, tmp_traj_V, 100, cmap='jet')
    plt.colorbar()
    sample_freq = 5
    plt.quiver(X1[::sample_freq, ::sample_freq], X2[::sample_freq, ::sample_freq], force[::sample_freq, ::sample_freq, 0], force[::sample_freq, ::sample_freq, 1])
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plt.savefig(os.path.join(save_path, f'em_potential_only_diffusion_{experiment_idx}.png'))
    plt.close()
