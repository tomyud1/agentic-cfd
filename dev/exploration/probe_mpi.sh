#!/bin/bash
source /usr/lib/openfoam/openfoam2606/etc/bashrc >/dev/null 2>&1
which mpirun; mpirun --version 2>&1 | head -2; echo "WM_MPLIB=$WM_MPLIB FOAM_MPI=$FOAM_MPI"; nproc
ls $FOAM_TUTORIALS/incompressible/pimpleFoam/laminar/ ; ls -d $FOAM_TUTORIALS/*/*/cylinder* $FOAM_TUTORIALS/*/*/*/cylinder* 2>/dev/null
grep -m3 "No Iterations" $(ls -d /home/tomyud/OpenFOAM/tomyud-v2606/run/agentic/stokes_first_problem/N2000_dt2.5ms)/log.pimpleFoam
grep -m2 "ExecutionTime" $(ls -d /home/tomyud/OpenFOAM/tomyud-v2606/run/agentic/stokes_first_problem/N2000_dt2.5ms)/log.pimpleFoam
