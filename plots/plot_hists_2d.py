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

def compute_hist_UF2024(ID_min = H1, energyrange=[1., 3.], bins=50):
    E_max = 26. * np.power(10., 18.6 - 20.) # 10^20 eV
    NDISTANCES = 100

    hists = []
    
    # hist_0, bin_edges = get_hist(f'sims/{model}_source_100000.txt', cutoff_energy)

    for i in range(NDISTANCES):
        filename = f'sims/crpropa_events_Fe_{i}_100000.txt'
        ID, E, E_source = np.loadtxt(filename, unpack=True, usecols=(2, 3, 6))

        idx = np.where(ID >= ID_min)
        ID = ID[idx]
        E = E[idx] / 1e2
        E_source = E_source[idx] / 1e2

        print(f'processing {i}')
        print(f'E range : {min(E):.3f} - {max(E):.3f} 10^20 eV')
        #print(f'ID range : {min(ID)} - {max(ID)}')
        print(f'size : {len(E)/1000} k')

        w = np.exp(-E_source / E_max)
        hist, bin_edges = np.histogram(E, weights=w, bins=bins, range=energyrange)
        hists.append(hist)

        print(f'hist : {min(hist):.3f} - {max(hist):.3f}')

    # Normalize histograms
    hists = np.array(hists)
    hists /= hists[0]
    np.savetxt('hists_2d_UF2024.txt', hists)

def compute_hist_CF_Iron(ID_min = H1, energyrange=[1., 3.], bins=50):
    E_0 = 1e-2 # 10^20 eV
    gamma = -1.47
    E_max = 26 * np.power(10., 18.19 - 20.) # 10^20 eV
    
    NDISTANCES = 100

    hists = []
    
    # hist_0, bin_edges = get_hist(f'sims/{model}_source_100000.txt', cutoff_energy)

    for i in range(NDISTANCES):
        filename = f'sims/crpropa_events_Fe_{i}_100000.txt'
        ID, E, E_source = np.loadtxt(filename, unpack=True, usecols=(2, 3, 6))

        idx = np.where(ID >= ID_min)
        ID = ID[idx]
        E = E[idx] / 1e2
        E_source = E_source[idx] / 1e2

        print(f'processing {i}')
        print(f'E range : {min(E):.3f} - {max(E):.3f} 10^20 eV')
        #print(f'ID range : {min(ID)} - {max(ID)}')
        print(f'size : {len(E)/1000} k')

        w = [(_E / E_0) ** (-gamma + 1) * (np.exp(1 - _E / E_max) if _E >= E_max else 1) for _E in E_source]
        hist, bin_edges = np.histogram(E, weights=w, bins=bins, range=energyrange)
        hists.append(hist)

        print(f'hist : {min(hist):.3f} - {max(hist):.3f}')

    # Normalize histograms
    hists = np.array(hists)
    hists /= hists[0]
    np.savetxt('hists_2d_CF_Iron.txt', hists)

def compute_hist_CF(ID_min = H1, energyrange=[1., 3.], bins=50):
    def get_weights(filename, L, Z, ID_min = H1):
        ID, E, E_source = np.loadtxt(filename, unpack=True, usecols=(2, 3, 6))
        
        idx = np.where(ID >= ID_min)
        ID = ID[idx]
        E = E[idx] / 1e2
        E_source = E_source[idx] / 1e2
        
        E_0 = 1e-2 # 10^20 eV
        gamma = -1.47
        E_min = np.power(10., 17.8 - 20.) # 10^20 eV
        E_cut = Z * np.power(10., 18.19 - 20.) # 10^20 eV        
        I = 1. / (2. - gamma) * (np.power(E_cut / E_0, 2. - gamma) - np.power(E_min / E_0, 2. - gamma))
        w_0 = L / E_0**2 / I

        w = [w_0 * (_E / E_0) ** (-gamma + 1) * (np.exp(1 - _E / E_cut) if _E >= E_cut else 1) for _E in E_source]

        return E, w

    NDISTANCES = 100

    hists = []
    
    # hist_0, bin_edges = get_hist(f'sims/{model}_source_100000.txt', cutoff_energy)
      
    for i in range(NDISTANCES):
        filename = f'sims/crpropa_events_He_{i}_100000.txt'
        E_He, w_He = get_hist(filename, 0.245, 2)
        filename = f'sims/crpropa_events_N_{i}_100000.txt'
        E_N, w_N = get_hist(filename, 0.681, 7)
        filename = f'sims/crpropa_events_Si_{i}_100000.txt'
        E_Si, w_Si = get_hist(filename, 0.049, 14)
        filename = f'sims/crpropa_events_Fe_{i}_100000.txt'
        E_Fe, w_Fe = get_hist(filename, 0.025, 26)

        E = np.concatenate((E_He, E_N, E_Si, E_Fe))
        w = np.concatenate((w_He, w_N, w_Si, w_Fe))

        print(f'processing {i}')
        print(f'E range : {min(E):.3f} - {max(E):.3f} 10^20 eV')
        #print(f'ID range : {min(ID)} - {max(ID)}')
        print(f'size : {len(E)/1000} k')
      
        hist, _ = np.histogram(E, weights=w, bins=bins, range=energyrange)
        hists.append(hist)

        print(f'hist : {min(hist):.3f} - {max(hist):.3f}')

    # Normalize histograms
    hists = np.array(hists)
    hists /= hists[0]
    np.savetxt('hists_2d_CF.txt', hists)

    # ID, E, E_source = np.loadtxt(filename, unpack=True, usecols=(2, 3, 6))
    # i = np.where(ID >= ID_min)
    # ID = ID[i]
    # E = E[i] / 1e2
    # E_source = E_source[i] / 1e2

    # w = np.exp(-E_source / E_max)
    
    # print(f'E range : {min(E)} - {max(E)} 10^20 eV')
    # print(f'ID range : {min(ID)} - {max(ID)}')
    # print(f'size : {len(E)/1000} k')

    # hist, bin_edges = np.histogram(E, weights=w, bins=bins, range=range)
    # return hist, bin_edges

# def get_hist(filename, Z, ID_min = H1, range=[1., 3.], bins=50):
#     ID, E, E_source = np.loadtxt(filename, unpack=True, usecols=(2, 3, 6))
#     i = np.where(ID >= ID_min)
#     ID = ID[i]
#     E = E[i] / 1e2
#     E_source = E_source[i] / 1e2

#     E_0 = 1e-2 # 10^20 eV
#     gamma = -1.47
#     E_cut = 26 * np.power(10., 18.19 - 20.) # 10^20 eV
#     w = [(_E / E_0) ** (-gamma + 1) * (np.exp(1 - _E / E_cut) if _E >= E_cut else 1) for _E in E_source]
    
#     print(f'E range : {min(E)} - {max(E)} 10^20 eV')
#     print(f'ID range : {min(ID)} - {max(ID)}')
#     print(f'size : {len(E)/1000} k')

#     hist, bin_edges = np.histogram(E, weights=w, bins=bins, range=range)
#     return hist, bin_edges

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

def plot_hists_2D(model = 'hists_2d_UF2024', figname = 'UF2024_Fe'):
    fig, ax = plt.subplots(figsize=(13.5, 8.5), dpi=300)  # High DPI for better resolution
    set_axes(ax, 'Energy [$10^{20}$ eV]', 'Distance [Mpc]', xscale='linear', yscale='log', 
             xlim=[1., 3.], ylim=[1., 200.])

    NDISTANCES = 100
    distances = np.logspace(np.log10(1.), np.log10(200.), NDISTANCES)

    NENERGIES = 50
    energies = np.linspace(1., 3., NENERGIES)

    heatmap_data = np.loadtxt(f'{model}.txt')

    horizon = []
    for i in range(NENERGIES):
        y = heatmap_data[:, i]
        ih = get_horizon(y)
        assert(ih <= NDISTANCES)
        horizon.append(distances[ih])

    print(f'horizon: {min(horizon):.3f} - {max(horizon):.3f} Mpc')

    ax.plot(energies, horizon, lw=3, color='r')

    c = ax.pcolormesh(energies, distances, heatmap_data, vmin=0.1, vmax=1.5, cmap='Greys', shading='auto')
    fig.colorbar(c, ax=ax, label='Normalized Count')
    #ax.text(2.1, 100., r'Fe $\rightarrow$ all', fontsize=29, color='k')

    add_pao(ax, r'PAO191110', 1.66, 0.13, 'tab:blue')
    add_pao(ax, r'PAO150825', 1.25, 0.10, 'gold')

    plt.tight_layout()  # Ensures better layout
    savefig(fig, f'{figname}.pdf')

def compute_hist():
    compute_hist_UF2024()
    compute_hist_CF_Iron()
    compute_hist_CF()

if __name__ == '__main__':
    plot_hists_2D('hists_2d_UF2024', 'hists_2d_UF2024')
    plot_hists_2D('hists_2d_CF_Iron', 'hists_2d_CF_Iron')
    plot_hists_2D('hists_2d_CF', 'hists_2d_CF')
