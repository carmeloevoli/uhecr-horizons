def source(energy_range, id_nucles, size=10000):
    energy_min, energy_max = energy_range
    A, Z = id_nucles
   
    filename = f'crpropa_events_{A}_{Z}_source_{size}.txt'

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

#    # Source
#    source = Source()
#    source.add(SourcePosition(10. * kpc))
#    source.add(SourceRedshift1D())

    # Source
    source = Source()
    source.add(SourcePosition(10. * kpc))
    source.add(SourceRedshift1D())
    source.add(SourcePowerLawSpectrum(energy_min, energy_max, -1))
    source.add(SourceParticleType(nucleusId(A, Z)))
    
    # Power law spectrum
    # composition = SourceComposition(energy_min, energy_max, -1)
    # composition.add(A, Z, 1)
    # source.add(composition)

    # Run simulation
    sim.setShowProgress(True)
    sim.run(source, size, True)
    output.close()
