import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from utils import set_axes, savefig, nucleusID
from load_sim import load_sim

# Configure Matplotlib backend
matplotlib.use('MacOSX')
plt.style.use('simprop.mplstyle')

# Nuclei
H1 = nucleusID(1, 1)
He4 = nucleusID(2, 2)
N14 = nucleusID(7, 14)
Si28 = nucleusID(14, 28)
Fe56 = nucleusID(26, 56)

NDISTANCES = 100

def compute_hist_UF2024(ID_min = H1, energyrange=[1., 3.], bins=50):
    hists = []
    
    for i in range(NDISTANCES):
        filename = f'sims/crpropa_events_Fe_{i}_100000.txt'
        ID_obs, E_obs, Z_obs, w_uf24, w_cf = load_sim(filename, 26, 1, ID_min=ID_min)

        print(f'processing {i}')
        print(f'E range : {min(E_obs):.3f} - {max(E_obs):.3f} 10^20 eV')
        #print(f'ID range : {min(ID)} - {max(ID)}')
        print(f'size : {len(E_obs)/1000} k')

        hist, bin_edges = np.histogram(E_obs, weights=w_uf24, bins=bins, range=energyrange)
        hists.append(hist)

        print(f'hist : {min(hist):.3f} - {max(hist):.3f}')

    # Normalize histograms
    hists = np.array(hists)
    hists /= hists[0]
    np.savetxt('hists_2d_UF2024.txt', hists)

def compute_hist_CF(ID_min = H1, energyrange=[1., 3.], bins=50):
    hists = []
          
    for i in range(NDISTANCES):
        filename = f'sims/crpropa_events_He_{i}_100000.txt'
        _, E_He, _, _, w_He = load_sim(filename, 2, 0.245, ID_min=ID_min)
        filename = f'sims/crpropa_events_N_{i}_100000.txt'
        _, E_N, _, _, w_N = load_sim(filename, 7, 0.681, ID_min=ID_min)
        filename = f'sims/crpropa_events_Si_{i}_100000.txt'
        _, E_Si, _, _, w_Si = load_sim(filename, 14, 0.049, ID_min=ID_min)
        filename = f'sims/crpropa_events_Fe_{i}_100000.txt'
        _, E_Fe, _, _, w_Fe = load_sim(filename, 26, 0.025, ID_min=ID_min)

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
    compute_hist_CF()

if __name__ == '__main__':
    compute_hist()
    plot_hists_2D('hists_2d_UF2024', 'hists_2d_UF2024')
    plot_hists_2D('hists_2d_CF', 'hists_2d_CF')
