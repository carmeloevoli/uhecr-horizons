import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from utils import set_axes, savefig, nucleusID

# Configure Matplotlib backend
matplotlib.use('MacOSX')
plt.style.use('simprop.mplstyle')

# Nuclei
H1 = nucleusID(1, 1)
He4 = nucleusID(2, 2)
N14 = nucleusID(7, 14)
Si28 = nucleusID(14, 28)
Fe56 = nucleusID(26, 56)

def get_hist(filename, E_max, ID_min = H1, range=[1., 3.], bins=50):
    ID, E, E_source = np.loadtxt(filename, unpack=True, usecols=(2, 3, 6))
    i = np.where(ID >= ID_min)
    ID = ID[i]
    E = E[i] / 1e2
    E_source = E_source[i] / 1e2

    w = np.exp(-E_source / E_max)
    
    print(f'E range : {min(E)} - {max(E)} 10^20 eV')
    print(f'ID range : {min(ID)} - {max(ID)}')
    print(f'size : {len(E)/1000} k')

    hist, bin_edges = np.histogram(E, weights=w, bins=bins, range=range)
    return hist, bin_edges

def get_horizon(attenuation_factor):
    size = len(attenuation_factor)
    for i in range(size):
        if attenuation_factor[i] < 0.1:
            return i

def add_pao(ax, name, energy, denergy, color):
    ax.axvline(energy, color=color, linestyle='-', linewidth=2.5)
    ax.axvline(energy - denergy, color=color, linestyle='--', linewidth=2.5)
    ax.axvline(energy + denergy, color=color, linestyle='--', linewidth=2.5)
    ax.fill_betweenx([0.5, 300.], energy - denergy, energy + denergy, color=color, alpha=0.1)
    ax.text(energy - 0.04, 100., name, fontsize=19, color=color, ha='center', va='center', rotation=90)

def plot_hists_2D(ID_min = H1, model = 'crpropa_events_56_26', figname = 'UF2024'):
    fig, ax = plt.subplots(figsize=(13.5, 8.5), dpi=300)  # High DPI for better resolution
    set_axes(ax, 'Energy [$10^{20}$ eV]', 'Distance [Mpc]', xscale='linear', yscale='log', 
             xlim=[1., 3.], ylim=[1., 200.])

    NDISTANCES = 100
    distances = np.logspace(np.log10(1.), np.log10(200.), NDISTANCES)

    NENERGIES = 50
    energies = np.linspace(1., 3., NENERGIES)

    cutoff_energy = np.power(10., 18.6 - 20.) * 26 # 10^20 eV

    heatmap_data = np.zeros((NDISTANCES, NENERGIES))

    hist_0, bin_edges = get_hist(f'sims/{model}_source_100000.txt', cutoff_energy)

    for i in range(NDISTANCES):
        print(f'processing {i}')
        filename = f'sims/{model}_{i}_100000.txt'
        hist, bin_edges = get_hist(filename, cutoff_energy, ID_min=ID_min)
        hist /= hist_0
        print(f'hist : {min(hist):.3f} - {max(hist):.3f}')
        heatmap_data[i] = hist

    horizon = []
    for i in range(NENERGIES):
        y = heatmap_data[:, i]
        ih = get_horizon(y)
        assert(ih <= NDISTANCES)
        horizon.append(distances[ih])

    print(f'horizon: {min(horizon)} - {max(horizon)}')

    ax.plot(energies, horizon, lw=3, color='r')

    print(f'bin_edges size: {len(bin_edges)}, distances size: {len(distances)}')
    print(f'heatmap_data size: {heatmap_data.shape}')

    # ocean_r
    c = ax.pcolormesh(energies, distances, heatmap_data, vmin=0.1, vmax=1.5, cmap='Greys', shading='auto')
    fig.colorbar(c, ax=ax, label='Normalized Count')
    ax.text(2.1, 100., r'Fe $\rightarrow$ all', fontsize=29, color='k')

    add_pao(ax, r'PAO191110', 1.66, 0.13, 'tab:blue')
    add_pao(ax, r'PAO150825', 1.25, 0.10, 'gold')

    plt.tight_layout()  # Ensures better layout
    savefig(fig, figname)

if __name__ == '__main__':
    plot_hists_2D(H1, model = 'crpropa_events_Fe', figname = 'UF2024_Fe.pdf')
    
