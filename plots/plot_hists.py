import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from utils import set_axes, savefig

# Configure Matplotlib backend
matplotlib.use('MacOSX')
plt.style.use('simprop.mplstyle')

# H = 1000010010
# He = 1000020040
# C = 1000060120
# N = 1000070140
# O = 1000080160
# Si = 1000140280  
# Fe = 1000260560

def get_hist(filename, E_max, range=[1, 4], bins=100):
    ID, E, E_source = np.loadtxt(filename, unpack=True, usecols=(2, 3, 6))
    w = np.exp(-E_source / E_max)
    
    print(f'E range : {min(E)} - {max(E)} EeV')
    print(f'size : {len(E)/1000} k')

    hist, bin_edges = np.histogram(np.log10(E), weights=w, bins=100, range=range)
    return hist, bin_edges

def plot_hists():
    def plot_single_hist(i, E_max, color):
        distance = np.logspace(np.log10(1.), np.log10(300.), 300)[i]
        hist, bin_edges = get_hist(f'sims/crpropa_events_56_26_{i}_10000.txt', E_max)
        ax.hist(bin_edges[:-1], bins=bin_edges, weights=hist, 
                lw=2.5, histtype='step', color=color, label=f'{distance:.1f} Mpc')

    fig, ax = plt.subplots(figsize=(13.5, 8.5))
    xlabel, ylabel = r'log$_{10}$ (E / EeV)', r'PDF'
    set_axes(ax, xlabel, ylabel, xscale='linear', yscale='log', xlim=[1, 3.5], ylim=[1e-2, 1e2])

    Z = 26.
    E_max = np.power(10., 0.6) * Z # EeV

    models = ((0, 'tab:olive'), 
              (50, 'tab:orange'), 
              (100, 'tab:red'), 
              (150, 'tab:blue'), 
              (200, 'tab:purple'))
    
    for i, color in models:
        plot_single_hist(i, E_max, color)

    ax.legend()
    savefig(fig, f'hists.pdf')

if __name__ == '__main__':
    plot_hists()
