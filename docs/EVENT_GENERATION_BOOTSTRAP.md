# H3-W10 Event Generation bootstrap

HADROS3 uses the micromamba environment `dis` for PYTHIA 8 and HepMC3.

```bash
micromamba install -n dis -c conda-forge pythia8=8.312 hepmc3=3.3.1
make hadros3-event-generator
micromamba run -n dis python scripts/event_generation/bootstrap_pythia.py
```

The installed HepMC package identifies itself as `3.03.01`. Its CMake package
is under `$CONDA_PREFIX/share/HepMC3/cmake`; the conda build does not provide a
`HepMC3.pc` file. `micromamba run -n dis` sets `PYTHIA8DATA`. The HADROS3
orchestrator also sets it explicitly for isolated jobs.

The accepted fixed-target policy is `PDF:lepton=off`, MPI off, ISR off, and FSR
with the POWHEG `SCALUP` veto. Generic ISR is retained as an experimental option
but fails the UHE four-momentum tolerance and is therefore not the default.
