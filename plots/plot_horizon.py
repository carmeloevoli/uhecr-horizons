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
C12 = nucleusID(6, 12)
N14 = nucleusID(7, 14)
Si28 = nucleusID(14, 28)
Fe56 = nucleusID(26, 56)

def get_hist(filename, E_source_cutoff, E_observed, dE_observed, ID_min, ID_max, range=[1., 3.], bins=100):
    ID, E, E_source = np.loadtxt(filename, unpack=True, usecols=(2, 3, 6))
    i = np.where((ID >= ID_min) & (ID <= ID_max))
    ID = ID[i]
    E = E[i] / 1e2 # 10^20 eV
    E_source = E_source[i] / 1e2 # 10^20 eV

    w = np.exp(-E_source / E_source_cutoff)
    w *= np.exp(-np.power((E - E_observed) / dE_observed, 2))
    
    hist, bin_edges = np.histogram(E, weights=w, bins=bins, range=range)
    return sum(hist)

def plot_horizon(figname = 'horizon'):
    fig, ax = plt.subplots(figsize=(13.5, 8.5), dpi=300)  # High DPI for better resolution
    set_axes(ax, 'Distance [Mpc]', 'attenuation', xscale='log', yscale='log', xlim=[1., 200.], ylim=[0.01, 2.])

    ndistances = 100
    distances = np.logspace(np.log10(1.), np.log10(200.), ndistances)

    filename = 'horizon_Fe_all.txt'
    horizon_lo, horizon_hi = np.loadtxt(filename, unpack=True, usecols=(1, 2))
    ax.plot(distances, horizon_hi, lw=3, color='tab:blue', label='PAO191110')
    ax.plot(distances, horizon_lo, lw=3, color='gold', label='PAO150825')

    filename = 'horizon_18.9_Si_all.txt'
    horizon_lo, horizon_hi = np.loadtxt(filename, unpack=True, usecols=(1, 2))
    ax.plot(distances, horizon_hi, lw=3, ls='--', color='tab:blue', label='PAO191110')
    ax.plot(distances, horizon_lo, lw=3, ls='--', color='gold', label='PAO150825')

    # filename = 'horizon_Fe_intermediate.txt'
    # horizon_lo, horizon_hi = np.loadtxt(filename, unpack=True, usecols=(1, 2))
    # ax.plot(distances, horizon_hi, lw=3, color='tab:blue', ls='--')
    # ax.plot(distances, horizon_lo, lw=3, color='gold', ls='--')

    ax.legend(fontsize=20, loc='upper right')

    plt.tight_layout()  # Ensures better layout
    savefig(fig, figname)

def compute_horizon(model, cutoff_energy, ID_min, ID_max, ndistances, txtname):
    horizon_lo = []
    norm_lo = get_hist(f'sims/{model}_source_100000.txt', cutoff_energy, 1.25, 0.10, ID_min=H1, ID_max=Fe56)
    
    horizon_hi = []
    norm_hi = get_hist(f'sims/{model}_source_100000.txt', cutoff_energy, 1.66, 0.13, ID_min=H1, ID_max=Fe56)    

    for i in range(ndistances):
        print(f'processing {i}')
        filename = f'sims/{model}_{i}_100000.txt'
        h = get_hist(filename, cutoff_energy, 1.25, 0.10, ID_min=ID_min, ID_max=ID_max)
        horizon_lo.append(h / norm_lo)
        h = get_hist(filename, cutoff_energy, 1.66, 0.13, ID_min=ID_min, ID_max=ID_max)
        horizon_hi.append(h / norm_hi)

    file = open(txtname, "w") 
    for i in range(ndistances): 
        file.write(f'{i} {horizon_lo[i]:e} {horizon_hi[i]:e}') 
        file.write("\n") 
    file.close() 

    print(f'Data is written into the file : {txtname}') 

if __name__ == '__main__':
    cutoff_energy = 26. * np.power(10., 18.6 - 20.) # 10^20 eV
    print(f'cutoff_energy = {cutoff_energy:.1f}')
    ndistances = 100
    # compute_horizon('crpropa_events_Fe', cutoff_energy, H1, Fe56, ndistances, 'horizon_18.6_Fe_all.txt')
    # compute_horizon('crpropa_events_Fe', cutoff_energy, H1, C12, ndistances, 'horizon_18.6_Fe_light.txt')
    # compute_horizon('crpropa_events_Fe', cutoff_energy, C12, Si28, ndistances, 'horizon_18.6_Fe_intermediate.txt')
    # compute_horizon('crpropa_events_Fe', cutoff_energy, Si28, Fe56, ndistances, 'horizon_18.6_Fe_heavy.txt')

    # cutoff_energy = 14. * np.power(10., 18.9 - 20.) # 10^20 eV
    # print(f'cutoff_energy = {cutoff_energy:.1f}')

    # compute_horizon('crpropa_events_Si', cutoff_energy, H1, Fe56, ndistances, 'horizon_18.9_Si_all.txt')
    # compute_horizon('crpropa_events_Si', cutoff_energy, H1, C12, ndistances, 'horizon_18.9_Si_light.txt')
    # compute_horizon('crpropa_events_Si', cutoff_energy, C12, Si28, ndistances, 'horizon_18.9_Si_intermediate.txt')
    # compute_horizon('crpropa_events_Si', cutoff_energy, Si28, Fe56, ndistances, 'horizon_18.9_Si_heavy.txt')

    plot_horizon(figname='horizon_Fe_all.pdf')

    # plot_horizon(H1, C12, model='crpropa_events_Fe', figname='horizon_Fe_light.pdf')    
    # plot_horizon(C12, Si28, model='crpropa_events_Fe', figname='horizon_Fe_intermediate.pdf') 
    # plot_horizon(Si28, Fe56, model='crpropa_events_Fe', figname='horizon_Fe_heavy.pdf')    
    # plot_horizon(H1, Fe56, model = 'crpropa_events_Fe', figname = 'horizon_Fe_all.pdf')
