function run() {
    note=rd/"$name":SNO."$seed"  bash run/.sh --pde reaction."$name" --model sno --mode 32 64 --spectral
    note=rd/"$name":NSM."$seed"  bash run/.sh --pde reaction."$name" --model fno --mode 32 64 --spectral
    note=rd/"$name"x64."$seed"   bash run/.sh --pde reaction."$name" --model fno --mode 32 64 --grid 64
    note=rd/"$name"x128."$seed"  bash run/.sh --pde reaction."$name" --model fno --mode 32 64 --grid 128
    note=rd/"$name"x256."$seed"  bash run/.sh --pde reaction."$name" --model fno --mode 32 64 --grid 256
    note=rd/"$name"x256C."$seed" bash run/.sh --pde reaction."$name" --model fno --mode 32 64 --grid 256 --cheb
    note=rd/"$name"x256F."$seed" bash run/.sh --pde reaction."$name" --model fno --mode 32 64 --spectral --fourier
}

for seed in 0 1 2 3; do

    seed=$seed name=nu005 run
    seed=$seed name=nu01 run
    seed=$seed name=nu05 iter=100000 run
    seed=$seed name=nu1 iter=200000 run

done