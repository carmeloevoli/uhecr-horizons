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

def get_hist(filename, E_source_cutoff, E_observed, dE_observed, ID_min = H1, range=[1., 3.], bins=100):
    ID, E, E_source = np.loadtxt(filename, unpack=True, usecols=(2, 3, 6))
    i = np.where(ID >= ID_min)
    ID = ID[i]
    E = E[i] / 1e2 # 10^20 eV
    E_source = E_source[i] / 1e2 # 10^20 eV

    w = np.exp(-E_source / E_source_cutoff)
    w *= np.exp(-np.power((E - E_observed) / dE_observed, 2))
    
    hist, bin_edges = np.histogram(E, weights=w, bins=bins, range=range)
    return sum(hist)

def plot_horizon(ID_min = H1, model = 'crpropa_events_56_26', figname = 'horizon'):
    fig, ax = plt.subplots(figsize=(13.5, 8.5), dpi=300)  # High DPI for better resolution
    set_axes(ax, 'Distance [Mpc]', 'attenuation', xscale='log', yscale='linear', xlim=[1., 200.], ylim=[0., 2.])

    ndistances = 100
    distances = np.logspace(np.log10(1.), np.log10(200.), ndistances)

    cutoff_energy = 26. * np.power(10., 18.6 - 20.) # 10^20 eV

    horizon_lo = []
    norm_lo = get_hist(f'sims/{model}_source_100000.txt', cutoff_energy, 1.25, 0.10, ID_min=ID_min)
    
    horizon_hi = []
    norm_hi = get_hist(f'sims/{model}_source_100000.txt', cutoff_energy, 1.66, 0.13, ID_min=ID_min)

    for i in range(ndistances):
        print(f'processing {i}')
        filename = f'sims/{model}_{i}_100000.txt'
        h = get_hist(filename, cutoff_energy, 1.25, 0.10, ID_min=ID_min)
        horizon_lo.append(h / norm_lo)
        h = get_hist(filename, cutoff_energy, 1.66, 0.13, ID_min=ID_min)
        horizon_hi.append(h / norm_hi)

    ax.plot(distances, horizon_hi, lw=3, color='tab:blue', label='PAO191110')
    ax.plot(distances, horizon_lo, lw=3, color='gold', label='PAO150825')

    ax.legend(fontsize=20, loc='upper right')

    plt.tight_layout()  # Ensures better layout
    savefig(fig, figname)

if __name__ == '__main__':
    plot_horizon(H1, model = 'crpropa_events_Fe', figname = 'horizon_Fe.pdf')
    