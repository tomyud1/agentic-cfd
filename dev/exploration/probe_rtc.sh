#!/bin/bash
source /usr/lib/openfoam/openfoam2606/etc/bashrc >/dev/null 2>&1
ls $FOAM_SRC/functionObjects/utilities/runTimeControl/runTimeCondition/ 2>/dev/null
ls /usr/lib/openfoam/openfoam2606/src/functionObjects/utilities/runTimeControl/runTimeCondition 2>/dev/null | head
sed -n 1,200p $FOAM_TUTORIALS/incompressible/simpleFoam/simpleCar/system/runTimeControls | grep -v "^ *//" | head -60
