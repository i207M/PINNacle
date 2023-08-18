ns_list=("NS_Long" "NS_NoObstacle" "LidDrivenFlow")
heat_list=("HeatMultiscale" "HeatComplex" "HeatLongTime" "HeatDarcy" "HeatND")
poi_list=("Poisson2D_Classic" "Poisson2D_hole" "Poisson3D" "Poisson2DManyArea" "PoissonND")
wave_list=("WaveEquation1D" "Wave2DLong" "WaveHetergeneous")
chao_list=("Kuramoto" "GrayScott")

for item in "${ns_list[@]}"
do
	python runs/run_all.py "$item" &
done
