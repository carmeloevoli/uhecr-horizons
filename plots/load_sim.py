import numpy as np
from utils import nucleusID, getChargeFromID

# Define nuclei ID constants
H1 = nucleusID(1, 1)
He4 = nucleusID(2, 2)
N14 = nucleusID(7, 14)
Si28 = nucleusID(14, 28)
Fe56 = nucleusID(26, 56)

GAMMA_CF = -1.47
# Constants in 10^20 eV units
RMAX_UF24 = 10 ** (18.6 - 20)
E_MIN_CF = 10 ** (17.8 - 20)
RCUT_CF = 10 ** (18.19 - 20)
E_0 = 1e-2  # 10^20 eV

def compute_weight_uf24(Z_source: float, E_source: np.ndarray) -> np.ndarray:
    return np.exp(-E_source / Z_source / RMAX_UF24)

def compute_weight_cf(Z_source: float, I_A: float, E_source: np.ndarray) -> np.ndarray:
    E_cut = Z_source * RCUT_CF
    integral = (1 / (2. - GAMMA_CF)) * ((E_cut / E_0) ** (2. - GAMMA_CF) - (E_MIN_CF / E_0) ** (2. - GAMMA_CF))
    w0 = I_A / E_0**2 / integral

    x = E_source / E_0
    suppression = np.where(E_source >= E_cut, np.exp(1 - E_source / E_cut), 1.0)
    return w0 * x ** (-GAMMA_CF + 1) * suppression

def load_sim(
    filename: str,
    Z_source: float,
    I_A: float,
    ID_min: int = H1,
    ID_max: int = Fe56
):
    ID_arr, E_arr, E_source_arr = np.loadtxt(filename, unpack=True, usecols=(2, 3, 6))

    # Filter by nucleus ID range
    valid_idx = (ID_arr >= ID_min) & (ID_arr <= ID_max)
    ID_obs = ID_arr[valid_idx]
    E_obs = E_arr[valid_idx] / 1e2 # Convert to 10^20 eV
    E_source_obs = E_source_arr[valid_idx] / 1e2 # Convert to 10^20 eV

    Z_obs = getChargeFromID(ID_obs)
    w_uf24 = compute_weight_uf24(Z_source, E_source_obs)
    w_cf = compute_weight_cf(Z_source, I_A, E_source_obs)

    return ID_obs, E_obs, Z_obs, w_uf24, w_cf
