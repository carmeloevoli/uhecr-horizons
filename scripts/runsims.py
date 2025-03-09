import numpy as np
from crpropa import *

NDISTANCES = 100

# Reference https://crpropa.github.io/CRPropa3/pages/example_notebooks/sim1D/sim1D.html
def run(i, energy_range, name_nucleus, id_nucles, size=10000):
    energy_min, energy_max = energy_range
    
    # Compute distance dynamically
    distance = np.logspace(np.log10(1.), np.log10(200.), NDISTANCES)[i] * Mpc
    filename = f'crpropa_events_{name_nucleus}_{i}_{size}.txt'
    
    print(f'fraction : {i} - D : {distance / Mpc:8.3f}')
    
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
    source.add(SourceParticleType(id_nucles))

    # Run simulation
    sim.setShowProgress(True)
    sim.run(source, size, True)
    output.close()

if __name__ == "__main__":
    energy_range = [1e2 * EeV, 1e4 * EeV]
    size = 100000
    
    nuclei = {
        "H": nucleusId(1, 1),
        "He": nucleusId(4, 2),
        "N": nucleusId(14, 7),
        "Si": nucleusId(28, 14),
        "Fe": nucleusId(56, 26),
    }

    for name, id in nuclei.items():
        for i in range(NDISTANCES):
            run(i, energy_range, name, id, size)
