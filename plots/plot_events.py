import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from utils import set_axes, savefig, nucleusID

# Configure Matplotlib backend
matplotlib.use('MacOSX')
plt.style.use('simprop.mplstyle')

def add_events(ax, x, y, x_err, color='tab:gray'):
    label='Auger data'
    zorder=1
    ax.errorbar(x, y, xerr=x_err, fmt='o', markeredgecolor=color, color=color, 
               label=label, capsize=4.5, markersize=8, elinewidth=2.2, capthick=2.2, zorder=zorder)
    ax.plot(x, y, 'o', markersize=7,color=color, zorder=zorder + 1)

def load_data(filename):
    energy, denergy, phi = np.loadtxt(filename, unpack=True, usecols=(2, 3, 5), delimiter=',', skiprows=1)
    energy = energy / 1e2
    denergy = denergy / 1e2
    phi = np.radians(phi) - np.pi
    sorted_indices = np.argsort(-energy)
    return energy[sorted_indices], denergy[sorted_indices], phi[sorted_indices]

def plot_events(filename = '../data/auger_catalog_SD.csv', figname = 'events.pdf'):
    fig, ax = plt.subplots(figsize=(13.5, 8.5), dpi=300)  # High DPI for better resolution
    set_axes(ax, r'E [$10^{20}$ eV]', '$\phi$ [radians]', xscale='linear', yscale='linear', xlim=[0.3, 1.9], ylim=[-np.pi, np.pi])

    # Load data
    energy, denergy, phi = load_data(filename)
    names = np.loadtxt(filename, dtype=str, delimiter=',', skiprows=1, usecols=(0,))
        
    # Plot data
    add_events(ax, energy, phi, denergy, 'lightgrey')

    # Add Fe max energy
    E_max = 26. * np.power(10., 0.6) # EeV
    ax.axvline(E_max / 1e2, color='tab:red', linestyle='--', linewidth=2.5, label='$26 R_{\rm cut}$ (UF2024)')
    ax.text(E_max / 1e2 - 0.03, -1.2, r'$26 R_{\rm cut}$ (UF2024)', fontsize=20, color='tab:red', ha='center', va='center', rotation=90)

    # Add Fe max energy (combined fit)
    E_max = 26. * np.power(10., 0.2) # EeV
    ax.axvline(E_max / 1e2, color='tab:orange', linestyle='--', linewidth=2.5, label='$26 R_{\rm cut}$ (combined fit)')
    ax.text(E_max / 1e2 - 0.03, -1.2, r'$26 R_{\rm cut}$ (combined fit)', fontsize=20, color='tab:orange', ha='center', va='center', rotation=90)

    colors = [
        'tab:blue', 'tab:purple', 'tab:cyan', 'tab:green', 'tab:olive',
        'tab:orange', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:red', 'gold'
    ]

    for i in range(11):
        print(f'{names[i]} {energy[i]:.2f} {denergy[i]:.2f} {phi[i]:.5f}')
        ax.text(energy[i], phi[i] - 0.2, f'{names[i]}', fontsize=14, color=colors[i], ha='center', va='center')
        add_events(ax, energy[i], phi[i], denergy[i], color=colors[i])

    plt.tight_layout()  # Ensures better layout
    savefig(fig, figname)

if __name__ == '__main__':
    plot_events()
