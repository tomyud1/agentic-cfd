# Dev experiment: faster steady convergence on case 2 N128 (not part of the case). One variant per call.
source /usr/lib/openfoam/openfoam2606/etc/bashrc
set -eu
SRC=$HOME/OpenFOAM/tomyud-v2606/run/agentic/lid_driven_cavity_re100
D=$HOME/OpenFOAM/tomyud-v2606/run/agentic/dev_steady
name=$1; kind=$2; par=$3; init=${4:-rest}; N=${5:-128}; steps=${6:-3000}
c=$D/$name; rm -rf $c; mkdir -p $c
if [ -d $SRC/N$N ]; then cp -r $SRC/N$N/system $SRC/N$N/constant $c/; cp -r $SRC/N$N/0 $c/0
else cp -r $SRC/N256/system $c/; mkdir -p $c/constant; cp $SRC/N256/constant/*Properties $c/constant/; cp -r $SRC/N256/0 $c/0
  sed -i "s/(256 256 1)/($N $N 1)/" $c/system/blockMeshDict; (cd $c && blockMesh > log.blockMesh 2>&1); fi
cd $c; if [ "$init" != rest ]; then mapFields $SRC/$init -consistent -sourceTime latestTime -case . > log.mapFields 2>&1; fi
cd $c
if [ $kind = simple ]; then
  sed -i "s/^\(\s*U\s*\)0\.9;/\1$par;/" system/fvSolution
  sed -i "/residualControl/,/}/d" system/fvSolution
  foamDictionary -entry endTime -set $steps system/controlDict > /dev/null
  app=simpleFoam
else
  dt=$(python3 -c "print($par/$N)")
  sed -i "s/default\s*steadyState;/default Euler;/" system/fvSchemes
  cat > system/fvSolution <<EOD
FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }
solvers
{
    p      { solver GAMG; smoother GaussSeidel; tolerance 1e-12; relTol 0.05; }
    pFinal { \$p; relTol 0.05; }
    U      { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-12; relTol 0.1; }
    UFinal { \$U; }
}
PIMPLE { nOuterCorrectors 1; nCorrectors 2; nNonOrthogonalCorrectors 0; momentumPredictor yes; pRefCell 0; pRefValue 0; }
EOD
  foamDictionary -entry deltaT -set $dt system/controlDict > /dev/null
  foamDictionary -entry endTime -set $(python3 -c "print($steps*$par/$N)") system/controlDict > /dev/null
  foamDictionary -entry writeControl -set timeStep system/controlDict > /dev/null
  foamDictionary -entry writeInterval -set 100000 system/controlDict > /dev/null
  foamDictionary -entry application -set pimpleFoam system/controlDict > /dev/null
  app=pimpleFoam
fi
t0=$(date +%s.%N); $app > log.$app 2>&1 || echo "$name FAILED"; t1=$(date +%s.%N)
echo "$name $(python3 -c "print(round($t1-$t0,1))") s" >> $D/times.txt
