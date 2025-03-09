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

def getChargeFromID(ID):
    return np.floor((ID / 10000) % 1000)

def get_hist(filename, E_source_cutoff, E_observed, dE_observed, ID_min = H1):
    ID, E, E_source = np.loadtxt(filename, unpack=True, usecols=(2, 3, 6))
    i = np.where(ID >= ID_min)
    ID = ID[i]
    E = E[i] # EeV
    E_source = E_source[i] # EeV
    Z = getChargeFromID(ID)
    print(f'Z = {min(Z)} - {max(Z)}')
    R = E / Z # EV

    w = np.exp(-E_source / E_source_cutoff)
    w *= np.exp(-np.power((E - E_observed) / dE_observed, 2))

    hist, bin_edges = np.histogram(R, weights=w, bins=50, range=[4., 12.])
    cdf = np.cumsum(hist)
    half_total = cdf[-1] / 2
    idx = np.searchsorted(cdf, half_total)
    median_R = 0.5 * (bin_edges[idx] + bin_edges[idx + 1])
    return hist, bin_edges, median_R

def plot_hist(ax, model, i, color):
    NDISTANCES = 100
    distances = np.logspace(np.log10(1.), np.log10(200.), NDISTANCES)
    label = f'{distances[i]:.0f} Mpc'

    cutoff_energy = np.power(10., 18.6 - 18.) * 26  # EV
    filename = f'sims/{model}_{i}_100000.txt'
    hist, bin_edges, R = get_hist(filename, cutoff_energy, 1.66e2, 0.13e2, ID_min=H1)
    print(f'processing {i} with distance {label} and mean rigidity {R:.3f}')
    ax.plot(bin_edges[:-1], hist, lw=3, color=color, label=label)
    ax.axvline(R, color=color, linestyle=':', linewidth=2.5)

def plot_rigidity(ID_min = H1, model = 'crpropa_events_56_26', figname = 'rigidity'):
    fig, ax = plt.subplots(figsize=(13.5, 8.5), dpi=300)  # High DPI for better resolution
    set_axes(ax, 'R [EV]', 'PDF', xscale='linear', yscale='linear', xlim=[4, 12], ylim=[0, 140]) # , xlim=[1., 4.], ylim=[0.5, 300.])

    plot_hist(ax, model, 1, 'tab:blue')
    plot_hist(ax, model, 43, 'gold')
    plot_hist(ax, model, 62, 'tab:red')

    ax.text(166. / 26., 130, 'Fe', color='tab:gray', fontsize=25, ha='center', va='center')
    ax.text(166. / 14., 130, 'Si', color='tab:gray', fontsize=25, ha='center', va='center')

    ax.legend(fontsize=20, loc='upper right')
    plt.tight_layout()  # Ensures better layout

    savefig(fig, figname)

if __name__ == '__main__':
    plot_rigidity(H1, model = 'crpropa_events_Fe', figname = 'rigidity.pdf')
    
