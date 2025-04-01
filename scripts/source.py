import numpy as np
from crpropa import *

def run(energy_range, name_nucleus, id_nucleus, size=10000):
    energy_min, energy_max = energy_range
   
    filename = f'crpropa_events_{name}_source_{size}.txt'

    # Simulation setup
    sim = ModuleList()
    sim.add(SimplePropagation(kpc, 10 * Mpc))

    # Observer and output
    obs = Observer()
    obs.add(Observer1D())
    output = TextOutput(filename, Output.Event1D)
    output.enable(output.SerialNumberColumn)
    obs.onDetection(output)
    sim.add(obs)

    # Source
    source = Source()
    source.add(SourcePosition(10. * kpc))
    source.add(SourceRedshift1D())
    source.add(SourcePowerLawSpectrum(energy_min, energy_max, -1))
    source.add(SourceParticleType(id_nucleus))
    
    # Run simulation
    sim.setShowProgress(True)
    sim.run(source, size, True)
    output.close()

if __name__ == "__main__":
    energy_range = [1e2 * EeV, 1e3 * EeV]
    size = 1000000
    
    nuclei = {
        "H": nucleusId(1, 1),
        "He": nucleusId(4, 2),
        "N": nucleusId(14, 7),
        "Si": nucleusId(28, 14),
        "Fe": nucleusId(56, 26),
    }

    for name, id in nuclei.items():
        run(energy_range, name, id, size)
