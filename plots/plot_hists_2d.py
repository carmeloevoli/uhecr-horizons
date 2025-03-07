import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from utils import set_axes, savefig, nucleusID

# Configure Matplotlib backend
matplotlib.use('MacOSX')
plt.style.use('simprop.mplstyle')

# Improve font rendering for PDFs
plt.rcParams['pdf.fonttype'] = 42  
plt.rcParams['savefig.dpi'] = 300  # High resolution

# Nuclei
H1 = nucleusID(1, 1)
He4 = nucleusID(2, 2)
N14 = nucleusID(7, 14)
Si28 = nucleusID(14, 28)
Fe56 = nucleusID(26, 56)

def get_hist(filename, E_max, ID_min = H1, range=[1e2, 4e2], bins=50):
    ID, E, E_source = np.loadtxt(filename, unpack=True, usecols=(2, 3, 6))
    i = np.where(ID >= ID_min)
    ID = ID[i]
    E = E[i]
    E_source = E_source[i]

    w = np.exp(-E_source / E_max)
    
 #   print(f'E range : {min(E)} - {max(E)} EeV')
    print (f' ID range : {min(ID)} - {max(ID)}')
 #   print(f'size : {len(E)/1000} k')

    hist, bin_edges = np.histogram(E, weights=w, bins=bins, range=range)
    return hist, bin_edges

def get_horizon(attenuation_factor):
    size = len(attenuation_factor)
    for i in range(size):
        if attenuation_factor[i] < 0.1:
            return i

def plot_hists_2D(ID_min = H1, figname = 'UF2024'):
    fig, ax = plt.subplots(figsize=(13.5, 8.5), dpi=300)  # High DPI for better resolution
    set_axes(ax, 'Energy [EeV]', 'Distance [Mpc]', xscale='linear', yscale='log', xlim=[1., 4.], ylim=[0.5, 300.])

    ndistances = 300
    distances = np.logspace(np.log10(0.1), np.log10(300.), ndistances)

    nenergies = 50
    energies = np.linspace(1., 4., nenergies)

    cutoff_energy = np.power(10., 0.6) * 26  # EeV

    heatmap_data = np.zeros((ndistances, nenergies))

    hist_0, bin_edges = get_hist(f'sims/crpropa_events_56_26_source_10000.txt', cutoff_energy)

    for i in range(ndistances):
        print(f'processing {i}')
        filename = f'sims/crpropa_events_56_26_{i}_10000.txt'
        hist, bin_edges = get_hist(filename, cutoff_energy, ID_min=ID_min)
        print(min(hist), max(hist))
        heatmap_data[i] = hist

    heatmap_data /= hist_0
    max_value = np.max(heatmap_data)

    horizon = []
    for i in range(nenergies):
        y = heatmap_data[:, i]
        ih = get_horizon(y)
        assert(ih <= ndistances)
        horizon.append(distances[ih])

    print(f'horizon: {horizon}')

    ax.plot(energies, horizon, lw=3, color='r')

    print(f'bin_edges size: {len(bin_edges)}, distances size: {len(distances)}, heatmap_data size: {heatmap_data.shape}')

    c = ax.pcolormesh(energies, distances, heatmap_data, vmin=0.1, vmax=1.5, cmap='ocean_r', shading='auto')
    fig.colorbar(c, ax=ax, label='Normalized Count')

    plt.tight_layout()  # Ensures better layout
    savefig(fig, figname)

if __name__ == '__main__':
    plot_hists_2D(H1, 'UF2024_all.pdf')
    
