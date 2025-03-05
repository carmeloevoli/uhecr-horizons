import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from utils import set_axes, savefig

# Configure Matplotlib backend
matplotlib.use('MacOSX')
plt.style.use('simprop.mplstyle')

def get_hist(filename, E_max, range=[1e2, 4e2], bins=100):
    ID, E, E_source = np.loadtxt(filename, unpack=True, usecols=(2, 3, 6))
    w = np.exp(-E_source / E_max)
    
    print(f'E range : {min(E)} - {max(E)} EeV')
    print(f'size : {len(E)/1000} k')

    hist, bin_edges = np.histogram(E, weights=w, bins=bins, range=range)
    return hist, bin_edges

def plot_hists_2D():
    fig, ax = plt.subplots(figsize=(13.5, 8.5))

    ndistances = 300
    distances = np.logspace(np.log10(0.1), np.log10(300.), ndistances)

    nenergies = 100
    energies = np.linspace(1., 4., nenergies)

    cutoff_energy = np.power(10., 0.6) * 26 # EeV

    heatmap_data = np.zeros((ndistances, nenergies))

    hist_0, bin_edges = get_hist(f'sims/crpropa_events_56_26_source_10000.txt', cutoff_energy)

    for i in range(ndistances):
        print (f'processing {i}')
        filename = f'sims/crpropa_events_56_26_{i}_10000.txt'
        hist, bin_edges = get_hist(filename, cutoff_energy)
        print(min(hist), max(hist))
        heatmap_data[i] = hist

    heatmap_data /= hist_0
    
    max_value = np.max(heatmap_data)
        
    print(f'bin_edges size: {len(bin_edges)}, distances size: {len(distances)}, heatmap_data size: {heatmap_data.shape}')

    c = ax.pcolormesh(energies, distances, heatmap_data, vmin=0.1 * max_value, vmax=max_value, cmap='turbo')
    fig.colorbar(c, ax=ax, label='Normalized Count')
    contour = ax.contour(energies, distances, heatmap_data, levels=[0.1 * max_value], colors='white')
    ax.set_yscale('log')
    ax.set_xlabel('Energy')
    ax.set_ylabel('distance')

    savefig(fig, f'UF2024.pdf')

if __name__ == '__main__':
    plot_hists_2D()
