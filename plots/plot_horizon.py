import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from utils import set_axes, savefig, nucleusID
from load_sim import load_sim
from scipy.signal import savgol_filter

# Configure Matplotlib backend
matplotlib.use('MacOSX')
plt.style.use('simprop.mplstyle')

# Nuclei
H1 = nucleusID(1, 1)
He4 = nucleusID(2, 2)
C12 = nucleusID(6, 12)
N14 = nucleusID(7, 14)
Si28 = nucleusID(14, 28)
Fe56 = nucleusID(26, 56)

NDISTANCES = 100

def compute_horizon_UF2024(E_event, dE_event, txtname):
    # def get_hist(filename, Z, E_observed, dE_observed, ID_min, ID_max, range=[1., 5.], bins=100):
    #     ID, E, E_source = np.loadtxt(filename, unpack=True, usecols=(2, 3, 6))
    #     i = np.where((ID >= ID_min) & (ID <= ID_max))
    #     ID = ID[i]
    #     E = E[i] / 1e2 # 10^20 eV
    #     E_source = E_source[i] / 1e2 # 10^20 eV
    #     E_cutoff = Z * np.power(10., 18.6 - 20.) # 10^20 eV

    #     w = np.exp(-E_source / E_cutoff)
    #     w *= np.exp(-np.power((E - E_observed) / dE_observed, 2))
    
    #     hist, bin_edges = np.histogram(E, weights=w, bins=bins, range=range)
    #     return np.sum(hist * np.diff(bin_edges))

    horizon = []

    for i in range(NDISTANCES):
        print(f'processing {i}')
        filename = f'sims/crpropa_events_Fe_{i}_100000.txt'
        ID_obs, E_obs, Z_obs, w_uf24, w_cf = load_sim(filename, 26, 1)
        w = w_uf24 * np.exp(-np.power((E_obs - E_event) / dE_event, 2))
        hist, bin_edges = np.histogram(E_obs, weights=w, bins=100, range=[1., 5.])
        h = np.sum(hist * np.diff(bin_edges))
        horizon.append(h)

    horizon = np.array(horizon)
    horizon /= horizon[0]

    np.savetxt(txtname, horizon)

    print(f'Data is written into the file : {txtname}') 

def compute_horizon_CF(E_event, dE_event, txtname):
    # def get_weights(filename, L, Z, E_observed, dE_observed, ID_min, ID_max):
    #     ID, E, E_source = np.loadtxt(filename, unpack=True, usecols=(2, 3, 6))
    #     idx = np.where((ID >= ID_min) & (ID <= ID_max))
    #     ID = ID[idx]
    #     E = E[idx] / 1e2
    #     E_source = E_source[idx] / 1e2
        
    #     E_0 = 1e-2 # 10^20 eV
    #     gamma = -1.47
    #     E_min = np.power(10., 17.8 - 20.) # 10^20 eV
    #     E_cut = Z * np.power(10., 18.19 - 20.) # 10^20 eV        
    #     I = 1. / (2. - gamma) * (np.power(E_cut / E_0, 2. - gamma) - np.power(E_min / E_0, 2. - gamma))
    #     w_0 = L / E_0**2 / I

    #     w = [w_0 * (_E / E_0) ** (-gamma + 1) * (np.exp(1 - _E / E_cut) if _E >= E_cut else 1) for _E in E_source]
    #     w *= np.exp(-np.power((E - E_observed) / dE_observed, 2))
    
    #     return E, w
    
    horizon = []

    for i in range(NDISTANCES):
        print(f'processing {i}')
        filename = f'sims/crpropa_events_He_{i}_100000.txt'
        _, E_He, _, _, w_He = load_sim(filename, 2, 0.245)
        filename = f'sims/crpropa_events_N_{i}_100000.txt'
        _, E_N, _, _, w_N = load_sim(filename, 7, 0.681)
        filename = f'sims/crpropa_events_Si_{i}_100000.txt'
        _, E_Si, _, _, w_Si = load_sim(filename, 14, 0.049)
        filename = f'sims/crpropa_events_Fe_{i}_100000.txt'
        _, E_Fe, _, _, w_Fe = load_sim(filename, 26, 0.025)

        E = np.concatenate((E_He, E_N, E_Si, E_Fe))
        w = np.concatenate((w_He, w_N, w_Si, w_Fe))
        w = w * np.exp(-np.power((E - E_event) / dE_event, 2))

        hist, bin_edges = np.histogram(E, weights=w, bins=100, range=[1., 5.])
        h = np.sum(hist * np.diff(bin_edges))

        horizon.append(h)

    horizon = np.array(horizon)
    horizon /= horizon[0]

    np.savetxt(txtname, horizon)

    print(f'Data is written into the file : {txtname}') 

def plot_horizon(figname = 'horizon'):
    fig, ax = plt.subplots(figsize=(13.5, 8.5), dpi=300)  # High DPI for better resolution
    set_axes(ax, 'Distance [Mpc]', 'attenuation', xscale='log', yscale='linear', xlim=[1., 200.], ylim=[0, 1.7])

    ndistances = 100
    distances = np.logspace(np.log10(1.), np.log10(200.), ndistances)

    horizon = np.loadtxt('horizon_UF24_1.7.txt')
    window_size = 5
    kernel = np.ones(window_size) / window_size
    #horizon = np.convolve(horizon, kernel, mode='same')

    horizon = savgol_filter(horizon, 21, 3, mode='nearest') # window size 51, polynomial order 3
    ax.plot(distances, horizon, lw=3, color='tab:blue', label='1.7')

    horizon = np.loadtxt('horizon_CF_1.7.txt')
    horizon = savgol_filter(horizon, 21, 3, mode='nearest') # window size 51, polynomial order 3
    ax.plot(distances, horizon, lw=3, color='tab:blue', label='1.7', ls='--')

    horizon = np.loadtxt('horizon_UF24_1.0.txt')
    horizon = savgol_filter(horizon, 21, 3, mode='nearest') # window size 51, polynomial order 3
    ax.plot(distances, horizon, lw=3, color='gold', label='1.0')

    horizon = np.loadtxt('horizon_CF_1.0.txt')
    horizon = savgol_filter(horizon, 21, 3, mode='nearest') # window size 51, polynomial order 3
    ax.plot(distances, horizon, lw=3, color='gold', label='1.0', ls='--')

    ax.legend(fontsize=20, loc='upper right')
 
    plt.tight_layout()  # Ensures better layout
    savefig(fig, figname)

if __name__ == '__main__':
    compute_horizon_UF2024(1.7, 0.08 * 1.7, txtname = 'horizon_UF24_1.7.txt')
    compute_horizon_UF2024(1.0, 0.08 * 1.0, txtname = 'horizon_UF24_1.0.txt')

    compute_horizon_CF(1.7, 0.08 * 1.7, txtname = 'horizon_CF_1.7.txt')
    compute_horizon_CF(1.0, 0.08 * 1.0, txtname = 'horizon_CF_1.0.txt')

    plot_horizon(figname='horizon_UF2024_vs_CF.pdf')

