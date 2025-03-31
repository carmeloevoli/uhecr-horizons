import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from utils import set_axes, savefig, nucleusID
from plot_combined_fit import get_source

# Configure Matplotlib backend
matplotlib.use('MacOSX')
plt.style.use('simprop.mplstyle')

def load_data(filename = '../data/HEFD_ICRC2023_v2_Id_sId_Neigh_POS_UTC_Th_Ph_RA_Dec_l_b_E'):
    data = np.loadtxt(filename, usecols=(1, 9, 10, 11))
    id = data[:, 0]
    l = data[:, 1]
    b = data[:, 2]
    energy = data[:, 3] / 1e2  # Convert to 10^20 eV
    sorted_indices = np.argsort(-energy)
    return energy[sorted_indices], b[sorted_indices], id[sorted_indices]

def add_events(ax, x, y, x_err, zorder, label, color='tab:gray'):
    ax.errorbar(x, y, xerr=x_err, fmt='o', markeredgecolor=color, color=color, 
               label=label, capsize=4.5, markersize=8, elinewidth=2.2, capthick=2.2, zorder=zorder)
    ax.plot(x, y, 'o', markersize=7,color=color, zorder=zorder + 1)

def plot_events(figname = 'events.pdf'):
    fig, ax = plt.subplots(figsize=(13.5, 8.5), dpi=300)  # High DPI for better resolution
    set_axes(ax, r'E [$10^{20}$ eV]', 'b [degrees]', xscale='linear', yscale='linear', xlim=[1, 2], ylim=[-90., 90.])

    # Load data
    energy, b, id = load_data()

    # Plot data
    add_events(ax, energy, b, 0.08 * energy, 1, 'ICRC2023', 'lightgrey')

    ax.text(1.9, -75, 'ICRC2023', fontsize=22, color='r', ha='center', va='center')

    N = np.count_nonzero(energy > 1)
    ax.text(1.83, 72, f'$N(>1)$ : {N}', fontsize=26, color='b', ha='center', va='center')

    colors = [
        'tab:blue', 'tab:purple', 'tab:cyan', 'tab:green', 'tab:olive',
        'tab:orange', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:red',
    ]

    for i in range(10):
        print(f'{id[i]} {energy[i]:.3f} {b[i]:.5f}')
        ax.text(energy[i], b[i] - 5, f'{int(id[i])}', fontsize=14, color=colors[i], ha='center', va='center')
        add_events(ax, energy[i], b[i], 0.08 * energy[i], 10, '', color=colors[i])

    plt.tight_layout()  # Ensures better layout
    savefig(fig, figname)

if __name__ == '__main__':
    plot_events()

    # # Add Fe max energy
    # E_max = 26. * np.power(10., 0.6) # EeV
    # ax.axvline(E_max / 1e2, color='tab:red', linestyle='--', linewidth=2.5, label='$26 R_{\rm cut}$ (UF2024)')
    # ax.text(E_max / 1e2 - 0.03, -1.2, r'$26 R_{\rm cut}$ (UF2024)', fontsize=20, color='tab:red', ha='center', va='center', rotation=90)

    # # Add Fe max energy (combined fit)
    # E_max = 26. * np.power(10., 0.2) # EeV
    # ax.axvline(E_max / 1e2, color='tab:orange', linestyle='--', linewidth=2.5, label='$26 R_{\rm cut}$ (combined fit)')
    # ax.text(E_max / 1e2 - 0.03, -1.2, r'$26 R_{\rm cut}$ (combined fit)', fontsize=20, color='tab:orange', ha='center', va='center', rotation=90)

    # # Add unpublished data
    # energy, phi = load_unpublished_data('../data/PAO_dataset_uhecr2024.txt')
    # #print (f'phi range : {min(phi):.2f}, {max(phi):.2f}')
    # #print (f'energy range : {min(energy):.2f}, {max(energy):.2f}')

    # color = 'k'
    # label = 'PAO 2024'
    # zorder = 3
    # ax.errorbar(energy, phi, xerr=0.08 * energy, fmt='o', markeredgecolor=color, color=color, 
    #            label=label, capsize=4.5, markersize=8, elinewidth=2.2, capthick=2.2, zorder=zorder)

# def load_data(filename):
#     energy, denergy, dec = np.loadtxt(filename, unpack=True, usecols=(2, 3, 7), delimiter=',', skiprows=1)
#     energy = energy / 1e2
#     denergy = denergy / 1e2
#     sorted_indices = np.argsort(-energy)
#     return energy[sorted_indices], denergy[sorted_indices], dec[sorted_indices]

# def load_unpublished_data(filename):
#     RA, dec, energy = np.loadtxt(filename, unpack=True, usecols=(0, 1, 2))
#     energy = energy / 1e2
#     i = np.where(energy > 1.25)
#     energy = energy[i]
#     dec = dec[i]
#     sorted_indices = np.argsort(-energy)
#     return energy[sorted_indices], dec[sorted_indices]

    # Plot data
    # add_events(ax, energy, b, 0.08 * energy, 1, 'lightgrey')

    # colors = [
    #     'tab:blue', 'tab:purple', 'tab:cyan', 'tab:green', 'tab:olive',
    #     'tab:orange', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:red', 'gold'
    # ]

    # for i in range(11):
    #     print(f'{id[i]} {energy[i]:.3f} {b[i]:.5f}')
    #     ax.text(energy[i], b[i] - 4, f'{int(id[i])}', fontsize=14, color=colors[i], ha='center', va='center')
    #     add_events(ax, energy[i], b[i], 0.08 * energy[i], 10, color=colors[i])
