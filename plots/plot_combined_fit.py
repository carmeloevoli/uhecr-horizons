import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from utils import set_axes, savefig

# Configure Matplotlib backend
matplotlib.use('MacOSX')
plt.style.use('simprop.mplstyle')

def source_spectrum(E, params):
    E_0 = 1e18 # eV
    L, gamma, E_min, E_cut = params
    I = 1. / (2. - gamma) * (np.power(E_cut / E_0, 2. - gamma) - np.power(E_min / E_0, 2. - gamma))
    Q_0 = L / E_0**2 / I
    print(f'Q0 : {Q_0:.2e}')
    Q = [Q_0 * (_E / E_0) ** (-gamma) * (np.exp(1 - _E/ E_cut) if _E >= E_cut else 1) for _E in E]
    return np.array(Q)

def get_source(E = np.logspace(17, 21, 1000)):
    # Define the parameters for the source spectrum
    E_min = np.power(10., 17.8) # eV
    R_cut = np.power(10., 18.19) # eV
    L_0 = 5e44 # erg/Mpc^3/yr
    L_0 *= 6.242e+11 # erg -> eV
    I_He = 0.245
    I_N = 0.681
    I_Si = 0.049
    I_Fe = 0.025
    gamma = -1.47

    Q_He = source_spectrum(E, [I_He * L_0, gamma, E_min, 2. * R_cut])
    Q_N = source_spectrum(E, [I_N * L_0, gamma, E_min, 7. * R_cut])
    Q_Si = source_spectrum(E, [I_Si * L_0, gamma, E_min, 14. * R_cut])
    Q_Fe = source_spectrum(E, [I_Fe * L_0, gamma, E_min, 26. * R_cut])

    mean_A = 4. * Q_He + 14. * Q_N + 28. * Q_Si + 56. * Q_Fe
    mean_A /= (Q_He + Q_N + Q_Si + Q_Fe)

    return E, Q_He, Q_N, Q_Si, Q_Fe, mean_A

def plot_combined_fit(figname='combined_fit.pdf'):
    fig, ax = plt.subplots(figsize=(13.5, 8.5), dpi=300)  # High DPI for better resolution
    set_axes(ax, 'log E', 'Q$_A$ [eV$^{-1}$ Mpc$^{-3}$ yr$^{-1}$]', xscale='linear', yscale='log', xlim=[17.8, 20.3] , ylim=[1e14, 1e20])

    # Load the source spectrum
    E, Q_He, Q_N, Q_Si, Q_Fe, A = get_source()

    # Plot the source spectrum
    logE = np.log10(E)
    ax.plot(logE, Q_He, color='tab:gray', label='He')
    ax.text(18, 1e19, 'He', fontsize=24, color='tab:gray', ha='center', va='center')
    ax.plot(logE, Q_N, color='tab:green', label='N')
    ax.text(18, 3e17, 'N', fontsize=24, color='tab:green', ha='center', va='center')
    ax.plot(logE, Q_Si, color='tab:cyan', label='Si')
    ax.text(18, 2.1e15, 'Si', fontsize=24, color='tab:cyan', ha='center', va='center')
    ax.plot(logE, Q_Fe, color='tab:blue', label='Fe')
    ax.text(18.5, 7e14, 'Fe', fontsize=24, color='tab:blue', ha='center', va='center')

    ax.fill_between([20, 20.3], 1e14, 1e20, color='tab:gray', alpha=0.15)

    #ax.legend(fontsize=20, loc='best')
    savefig(fig, figname)

def plot_combined_fit_highenergy(figname='combined_fit_highenergy.pdf'):
    fig, ax = plt.subplots(figsize=(13.5, 8.5), dpi=300)  # High DPI for better resolution
    set_axes(ax, 'E [$10^{20}$ eV]', '$Q_A$ [normalized]', xscale='linear', yscale='linear', xlim=[1, 6], ylim=[0, 1.05])

    # Load the source spectrum
    E, Q_He, Q_N, Q_Si, Q_Fe, A = get_source()
    Q_all = Q_He + Q_N + Q_Si + Q_Fe
    
    # Plot the source spectrum
    #ax.plot(E / 1e20, Q_He / Q_all, color='tab:gray', label='He / all')
    ax.plot(E / 1e20, Q_N / Q_all, color='tab:green', label='N / all')
    ax.plot(E / 1e20, Q_Si / Q_all, color='tab:cyan', label='Si / all')
    ax.plot(E / 1e20, Q_Fe / Q_all, color='tab:blue', label='Fe / all')

    ax.hlines(1, 1, 10, color='tab:gray', linestyle='--', lw=2)

    ax.legend(fontsize=20, loc='lower right')
    savefig(fig, figname)

def get_hist(filename, L, Z, bins=60):
    ID, E, E_source = np.loadtxt(filename, unpack=True, usecols=(2, 3, 6))
    E /= 1e2 # 10^20 eV
    E_source /= 1e2 # 10^20 eV

    E_0 = 1e-2
    gamma = -1.47
    E_min = np.power(10., 17.8 - 20.) # 10^20 eV
    E_cut = Z * np.power(10., 18.19 - 20.) # 10^20 eV
    I = 1. / (2. - gamma) * (np.power(E_cut / E_0, 2. - gamma) - np.power(E_min / E_0, 2. - gamma))
    w_0 = L / E_0**2 / I

    w = [w_0 * (_E / E_0) ** (-gamma + 1) * (np.exp(1 - _E / E_cut) if _E >= E_cut else 1) for _E in E_source]
    
    hist, bin_edges = np.histogram(E, weights=w, bins=100, range=[1, 10], density=True)
    print(np.sum(hist * np.diff(bin_edges)))
    return hist, bin_edges

def plot_combined_fit_histo(figname='combined_fit_histo.pdf'):
    fig, ax = plt.subplots(figsize=(13.5, 8.5), dpi=300)  # High DPI for better resolution
    set_axes(ax, 'E [$10^{20}$ eV]', 'dN/dE', xscale='linear', yscale='linear', xlim=[1, 3]) # , ylim=[1e-8, 1e1])

    # Load the source spectrum
    # E, Q_He, Q_N, Q_Si, Q_Fe, A = get_source(np.linspace(1e20, 10e20, 1000))

    # ax.plot(E / 1e20, Q_N, color='tab:green', label='N')
    # ax.plot(E / 1e20, Q_Si, color='tab:cyan', label='Si')
    # ax.plot(E / 1e20, Q_Fe, color='tab:blue', label='Fe')

    # Load the source histogram
    hist, bin_edges = get_hist('sims/crpropa_events_N_source_1000000.txt', 0.681, 7.)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    #ax.step(bin_centers, hist, where='mid', color='tab:green', label='N')
    ax.stairs(hist, bin_edges, label='N', color='tab:green', linewidth=3.0, hatch='//', alpha=0.9)

    hist, bin_edges = get_hist('sims/crpropa_events_Si_source_1000000.txt', 0.049, 14.)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    #ax.step(bin_centers, hist, where='mid', color='tab:cyan', label='Si')
    ax.stairs(hist, bin_edges, label='Si', color='tab:cyan', linewidth=3.0, hatch='\\', alpha=0.9)

    hist, bin_edges = get_hist('sims/crpropa_events_Fe_source_1000000.txt', 0.025, 26.)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    # ax.step(bin_centers, hist, where='mid', color='tab:blue', label='Fe')
    ax.stairs(hist, bin_edges, label='Fe', color='tab:blue', linewidth=3.0, hatch='|', alpha=0.9)

    #ax.legend(fontsize=20, loc='lower right')
    savefig(fig, figname)

if __name__ == '__main__':
    #plot_combined_fit() 
    plot_combined_fit_highenergy()
    #plot_combined_fit_histo()