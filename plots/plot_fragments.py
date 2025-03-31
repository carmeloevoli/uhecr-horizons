import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from utils import set_axes, savefig, nucleusID
from load_sim import load_sim, getChargeFromID

# Configure Matplotlib backend
matplotlib.use('MacOSX')
plt.style.use('simprop.mplstyle')

# Nuclei
H1 = nucleusID(1, 1)
He4 = nucleusID(2, 2)
N14 = nucleusID(7, 14)
Si28 = nucleusID(14, 28)
Fe56 = nucleusID(26, 56)

def get_fragments_UF2024(E_event, dE_event, ID_min = H1):
    filename = f'sims/crpropa_events_Fe_1.0_200.0_1000000.txt'
    ID_obs, E_obs, Z_obs, w_uf24, w_cf = load_sim(filename, 26, 1)
    w = w_uf24 * np.exp(-np.power((E_obs - E_event) / dE_event, 2))
    hist, bin_edges = np.histogram(Z_obs, weights=w, bins=26, range=[0.5, 26.5])
    return hist, bin_edges

def get_fragments_CF(E_event, dE_event, ID_min = H1):
    filename = f'sims/crpropa_events_He_1.0_200.0_1000000.txt'
    _, E_obs, Z_He, _, w_cf = load_sim(filename, 2, 0.245)
    w_He = w_cf * np.exp(-np.power((E_obs - E_event) / dE_event, 2))
    
    filename = f'sims/crpropa_events_N_1.0_200.0_1000000.txt'
    _, E_obs, Z_N, _, w_cf = load_sim(filename, 7, 0.681)    
    w_N = w_cf * np.exp(-np.power((E_obs - E_event) / dE_event, 2))
    
    filename = f'sims/crpropa_events_Si_1.0_200.0_1000000.txt'
    _, E_obs, Z_Si, _, w_cf = load_sim(filename, 14, 0.049)    
    w_Si = w_cf * np.exp(-np.power((E_obs - E_event) / dE_event, 2))
    
    filename = f'sims/crpropa_events_Fe_1.0_200.0_1000000.txt'
    _, E_obs, Z_Fe, _, w_cf = load_sim(filename, 26, 0.025)
    w_Fe = w_cf * np.exp(-np.power((E_obs - E_event) / dE_event, 2))

    Z = np.concatenate((Z_He, Z_N, Z_Si, Z_Fe))
    w = np.concatenate((w_He, w_N, w_Si, w_Fe))

    hist, bin_edges = np.histogram(Z, weights=w, bins=26, range=[0.5, 26.5])
    return hist, bin_edges

def plot_fragments(model = 'crpropa_events', figname = 'fragments.pdf'):
    fig, ax = plt.subplots(figsize=(13.5, 8.5), dpi=300)  # High DPI for better resolution
    set_axes(ax, 'Z', 'Probability', xscale='linear', yscale='linear', xlim=[0, 27], ylim=[0, 0.4])

    hist, bin_edges = get_fragments_UF2024(1.7, 0.08 * 1.7)
    widths = np.diff(bin_edges)
    ax.bar(bin_edges[:-1], hist / np.sum(hist), width=widths, align='edge', color='tab:blue', edgecolor='black', alpha=0.5, label='1.7')

    hist, bin_edges = get_fragments_UF2024(1.0, 0.08 * 1.0)
    widths = np.diff(bin_edges)
    ax.bar(bin_edges[:-1], hist / np.sum(hist), width=widths, align='edge', color='tab:olive', edgecolor='black', alpha=0.5, label='1.0')

    hist, bin_edges = get_fragments_CF(1.7, 0.08 * 1.7)
    x = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    hist = hist / np.sum(hist)
    ax.plot(x, hist, color='tab:blue', marker='o', markersize=6, label='1.7')

    hist, bin_edges = get_fragments_CF(1.0, 0.08 * 1.0)
    x = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    hist = hist / np.sum(hist)
    ax.plot(x, hist, color='tab:olive', marker='o', markersize=6, label='1.0')

    ax.legend(fontsize=20, loc='upper left')
    plt.tight_layout()  # Ensures better layout

    savefig(fig, figname)

if __name__ == '__main__':
    plot_fragments(model = 'crpropa_events_Fe', figname = 'fragments.pdf') 
