import matplotlib
import numpy as np
import matplotlib.pyplot as plt
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

# identify mass groups
# idx1 = A == 1
# idx2 = (A > 1) * (A <= 7)
# idx3 = (A > 7) * (A <= 28)
# idx4 = (A > 28)

def get_hist(filename, E_source_cutoff, ID_range, bins=60):
    ID, E, E_source = np.loadtxt(filename, unpack=True, usecols=(2, 3, 6))
    ID_min, ID_max = ID_range
    i = np.where((ID >= ID_min) * (ID <= ID_max))
    ID = ID[i]
    E = E[i] / 1e2 # 10^20 eV
    E_source = E_source[i] / 1e2 # 10^20 eV

    w = np.exp(-E_source / E_source_cutoff)
    
    hist, bin_edges = np.histogram(E, weights=w, bins=100, range=[1, 4])
    return hist, bin_edges

def plot_hists():
    fig, ax = plt.subplots(figsize=(13.5, 8.5))
    xlabel, ylabel = r'log$_{10}$ (E / EeV)', r'PDF'
    set_axes(ax, xlabel, ylabel, xscale='linear', yscale='log', xlim=[1, 4]) # , ylim=[1e-2, 1e3])

    E_max = np.power(10., 18.6 - 20.) * 26. # 10^20 eV

    hist_light = []
    hist_intermediate = []
    hist_heavy = []
    bin_edges = None

    NDISTANCES = 10

    for i in range(NDISTANCES):
        print(f'processing {i}')
        filename = f'sims/crpropa_events_Fe_{i}_100000.txt'
        hist, bin_edges = get_hist(filename, E_max, (H1, N14))
        hist_light.append(hist)

        hist, bin_edges = get_hist(filename, E_max, (H1, Si28))
        hist_intermediate.append(hist)

        hist, bin_edges = get_hist(filename, E_max, (H1, Fe56))
        hist_heavy.append(hist)

    hist_light = np.array(hist_light)
    hist_intermediate = np.array(hist_intermediate)
    hist_heavy = np.array(hist_heavy)

    ax.hist(bin_edges[:-1], bins=bin_edges, weights=np.mean(hist_light, axis=0), lw=2.5, histtype='stepfilled', alpha=0.25, color='r', label='H1-N14')
    ax.hist(bin_edges[:-1], bins=bin_edges, weights=np.mean(hist_intermediate, axis=0), lw=2.5, histtype='stepfilled', alpha=0.25, color='b', label='N14-Si28')
    ax.hist(bin_edges[:-1], bins=bin_edges, weights=np.mean(hist_heavy, axis=0), lw=2.5, histtype='stepfilled', alpha=0.25, color='g', label='Si28-Fe56') 

    ax.legend()
    savefig(fig, f'hists.pdf')

if __name__ == '__main__':
    plot_hists()
