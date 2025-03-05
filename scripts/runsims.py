import numpy as np
from crpropa import *

NDISTANCES = 300

def run(i, energy_range, id_nucles, size=10000):
    energy_min, energy_max = energy_range
    A, Z = id_nucles
    
    # Compute distance dynamically
    distance = np.logspace(np.log10(0.1), np.log10(300.), NDISTANCES)[i] * Mpc
    filename = f'crpropa_events_{A}_{Z}_{i}_{size}.txt'
    
    print(f'D : {distance / Mpc:8.3f}')
    
    # Simulation setup
    sim = ModuleList()
    sim.add(SimplePropagation(kpc, 10. * Mpc))
    sim.add(Redshift())
    sim.add(PhotoPionProduction(CMB()))
    sim.add(PhotoPionProduction(IRB_Saldana21()))
    sim.add(PhotoDisintegration(CMB()))
    sim.add(PhotoDisintegration(IRB_Saldana21()))
    sim.add(NuclearDecay())
    sim.add(ElectronPairProduction(CMB()))
    sim.add(ElectronPairProduction(IRB_Saldana21()))
    sim.add(MinimumEnergy(10 * EeV))

    # Observer and output
    obs = Observer()
    obs.add(Observer1D())
    output = TextOutput(filename, Output.Event1D)
    output.enable(output.SerialNumberColumn)
    obs.onDetection(output)
    sim.add(obs)

    # Source
    source = Source()
    source.add(SourcePosition(distance))
    source.add(SourceRedshift1D())
    source.add(SourcePowerLawSpectrum(energy_min, energy_max, -1))
    source.add(SourceParticleType(nucleusId(A, Z)))

    # Run simulation
    sim.setShowProgress(True)
    sim.run(source, size, True)
    output.close()

if __name__ == "__main__":
    energy_range = [1e2 * EeV, 1e4 * EeV]
    size = 10000
    
    nuclei = {
        "H": [1, 1],
        "He": [4, 2],
        "N": [14, 7],
        "Si": [28, 14],
        "Fe": [56, 26]
    }

    for name, nucleon in nuclei.items():
        for i in range(NDISTANCES):
            run(i, energy_range, nucleon, size)
