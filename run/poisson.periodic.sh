for seed in 0 1 2 3; do

    seed=$seed note=ps/relu/periodic:SNO."$seed" bash run/.sh --pde poisson.periodic --model sno --mode 31 31 --spectral
    seed=$seed note=ps/relu/periodic:NSM."$seed" bash run/.sh --pde poisson.periodic --model fno --mode 31 31 --spectral
    seed=$seed note=ps/relu/periodicx64."$seed"  bash run/.sh --pde poisson.periodic --model fno --mode 31 31 --grid 64
    seed=$seed note=ps/relu/periodicx128."$seed" bash run/.sh --pde poisson.periodic --model fno --mode 31 31 --grid 128
    seed=$seed note=ps/relu/periodicx256."$seed" bash run/.sh --pde poisson.periodic --model fno --mode 31 31 --grid 256

    seed=$seed note=ps/tanh/periodicx64."$seed"  bash run/.sh --pde poisson.periodic --model fno --mode 31 31 --grid 64  --activate tanh
    seed=$seed note=ps/tanh/periodicx128."$seed" bash run/.sh --pde poisson.periodic --model fno --mode 31 31 --grid 128 --activate tanh
    seed=$seed note=ps/tanh/periodicx256."$seed" bash run/.sh --pde poisson.periodic --model fno --mode 31 31 --grid 256 --activate tanh

    iter=100000 \
    seed=$seed note=ps/long/periodicx256."$seed" bash run/.sh --pde poisson.periodic --model fno --mode 31 31 --grid 256 --activate tanh

done