import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from utils import set_axes, savefig, nucleusID
from load_sim import load_sim, getChargeFromID
import pandas as pd
from scipy.signal import savgol_filter

# Configure Matplotlib backend
matplotlib.use('MacOSX')
plt.style.use('simprop.mplstyle')

MINWEIGHT = 1e-3

def get_db(filename, catalog = 'Swift-BAT'):
    df = pd.read_csv(
        filename, 
        delimiter='\t',
        skiprows=1, 
        names=["lon", "lat", "d", "name", "weight", "color", "label", "inside_contour"],
    )

    # Pick only where inside contour is True and weight > MINWEIGHT
    df = df[(df["inside_contour"] == True) & (df["weight"] > 1e-3)]

    # Pick only where label is 'catalog'
    df = df[df["label"] == catalog]

    print(f'catalog: {catalog}')
    print(f'long   : {df["lon"].min():.2f}, {df["lon"].max():.2f}')
    print(f'lat    : {df["lat"].min():.2f}, {df["lat"].max():.2f}')
    print(f'd      : {df["d"].min():.2f}, {df["d"].max():.2f}')
    print(f'weight : {df["weight"].min():.1e}, {df["weight"].max():.1e}')

    return df

def get_horizon(d, filename='horizon_UF24_1.0.txt'):
    distances = np.logspace(np.log10(1.), np.log10(200.), 100)
    attenuation = np.loadtxt(filename)
    return np.interp(d, distances, attenuation)

def _scatter(ax, df, markersize, label):
    w = df["weight"].values
    s = np.where((w > 1e-1) & (w < 1), 1400,
        np.where((w > 1e-2) & (w < 1e-1), 400,
            np.where((w > 1e-3) & (w < 1e-2), 100, 0)))

    d = df["d"].values
    h = get_horizon(d)

    print(min(h), max(h))
    norm = plt.Normalize(0.01, 1.0)

    sc = ax.scatter(
        df["lon"], df["lat"],
        s=s,
        c=h,
        cmap='coolwarm',
        norm=norm,
        alpha=0.4,
        marker=markersize,
        label=label,
        zorder=2
    )
    for lo, la in zip(df["lon"], df["lat"]):
        ax.plot([4.75, lo], [0, la], color='tab:gray', lw=1, alpha=0.5, zorder=1)

def plot_sources(figname = 'sources.pdf'):
    fig, ax = plt.subplots(figsize=(11.5, 8.5), dpi=300)  # High DPI for better resolution
    set_axes(ax, 'l', 'b', xscale='linear', yscale='linear', xlim=[4, 5.5], ylim=[-0.8, 0.8])
    
    filename = 'src_inside_ev_0.txt'
 
    # Scatter plot for Radio Galaxies
    df_radio = get_db(filename, catalog='Radio Galaxies')
    _scatter(ax, df_radio, 'o', 'Radio Galaxies')

    # Scatter plot for Swift-BAT
    df_swift = get_db(filename)
    _scatter(ax, df_swift, '*', 'Swift-BAT')
    
    # Add a colorbar for the scatter plot
    # cbar = plt.colorbar(sc, ax=ax)
    # cbar.set_label('Distance', fontsize=22)
    # cbar.ax.tick_params(labelsize=22)
    # cbar.set_ticks([0, 1, 2, 3])
    # cbar.set_ticklabels(['0', '1', '2', '3'])

    ax.text(4.8, 0.65, 'PAO20211911', fontsize=22, color='tab:blue', ha='center', va='center')

    # Add a legend with the label 'Event' for a circle and 'Radio Galaxies' for a star
    #ax.legend(['Radio Galaxies', 'Swift-BAT'], columnspacing=0.1, handles=[handle1, handle2], loc='upper right', fontsize=22, markerscale=0.8)   
    savefig(fig, figname)

def plot_attenuation(figname = 'attenuation.pdf'):
    fig, ax = plt.subplots(figsize=(11.5, 8.5), dpi=300)  # High DPI for better resolution
    set_axes(ax, 'd', 'a', xscale='log', yscale='linear', xlim=[1e0, 1e2]) # , ylim=[-0.8, 0.8])
    
    distances = np.logspace(np.log10(1.), np.log10(200.), 100)
    attenuation = np.loadtxt('horizon_UF24_1.0.txt')
    ax.plot(distances, attenuation, color='tab:gray', lw=4)

    filename = 'src_inside_ev_0.txt'
    df = get_db(filename, catalog='Radio Galaxies')
    d = df["d"].values
    ax.vlines(d, 0, 2, color='tab:red', lw=3, alpha=0.5)

    df = get_db(filename, catalog='Swift-BAT')  
    d = df["d"].values
    ax.vlines(d, 0, 2, color='tab:green', lw=3, alpha=0.5)

    df = get_db(filename, catalog='Lunardini')
    d = df["d"].values
    ax.vlines(d, 0, 2, color='tab:blue', lw=3, alpha=0.5)

    savefig(fig, figname)

def plot_sources_and_attenuation(figname = 'sources_and_attenuation.pdf'):
    fig, ax = plt.subplots(figsize=(11.5, 8.5), dpi=300)  # High DPI for better resolution
    set_axes(ax, 'distance [Mpc]', 'flux weights [\%]', xscale='linear', yscale='log', xlim=[5, 150], ylim=[1e-3, 1])
    
    filename = 'src_inside_ev_0.txt'

    import matplotlib.colors as colors

    df = get_db(filename, catalog='Radio Galaxies')
    w, d = df["weight"].values, df["d"].values    
    ax.scatter(d, w, s=300, c=w, norm=colors.LogNorm(vmin=3e-3, vmax=0.3), cmap='rainbow', alpha=1., marker='P', label='Radio Galaxies', zorder=2)

    df = get_db(filename, catalog='Swift-BAT')
    w, d = df["weight"].values, df["d"].values    
    ax.scatter(d, w, s=300, c=w, norm=colors.LogNorm(vmin=3e-3, vmax=0.3), cmap='rainbow', alpha=1., marker='X', label='Radio Galaxies', zorder=2)

    df = get_db(filename, catalog='Lunardini')
    w, d = df["weight"].values, df["d"].values    
    ax.scatter(d, w, s=300, c=w, norm=colors.LogNorm(vmin=3e-3, vmax=0.3), cmap='rainbow', alpha=1., marker='^', label='Radio Galaxies', zorder=2)

    distances = np.logspace(np.log10(1.), np.log10(200.), 100)
    attenuation = np.loadtxt('horizon_UF24_1.0.txt')
    attenuation = savgol_filter(attenuation, 21, 3, mode='nearest') # window size 51, polynomial order 3

    ax2 = ax.twinx()

    ax2.plot(distances, attenuation, color='tab:gray', lw=5, alpha=0.5)
    ax2.set_ylim(0, 1.5)
    ax2.set_yscale('linear')
    ax2.set_ylabel('Attenuation', fontsize=22, color='tab:gray')
    ax2.tick_params(axis='y', labelcolor='tab:gray')
    ax2.fill_between([125, 300], 0, 2, color='tab:gray', alpha=0.25)

    savefig(fig, figname)


if __name__ == "__main__":
    #plot_sources()  
    #plot_attenuation()
    #plot_sources_and_attenuation()
    
    # UTC = 1263041617
    
    from datetime import datetime

    # Given Unix timestamp
    timestamp = 1573399408

    # Convert to UTC datetime
    dt = datetime.utcfromtimestamp(timestamp)

    # Extract components
    year = dt.year
    month = dt.month
    day = dt.day

    print(f"UTC Date: {year}-{month:02d}-{day:02d} and PAO191110")

    # import datetime
    # import pytz
 
    # april_fools = datetime.datetime(2030, 4, 1, 10, 0)
    # print(april_fools.now(datetime.timezone.utc))

    # # Convert to UTC timezone
    # utc = pytz.utc
    # april_fools_utc = utc.localize(april_fools)
    # print(april_fools_utc)
 
 