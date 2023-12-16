for seed in 0 1 2 3; do

    seed=$seed iter=100000 note=ps/relu/dirichlet:SNO."$seed"  bash run/.sh --pde poisson.dirichlet --model sno --mode 31 31 --spectral
    seed=$seed iter=100000 note=ps/relu/dirichlet:NSM."$seed"  bash run/.sh --pde poisson.dirichlet --model fno --mode 31 31 --spectral
    seed=$seed iter=100000 note=ps/relu/dirichletx64."$seed"   bash run/.sh --pde poisson.dirichlet --model fno --mode 31 31 --grid 64
    seed=$seed iter=100000 note=ps/relu/dirichletx128."$seed"  bash run/.sh --pde poisson.dirichlet --model fno --mode 31 31 --grid 128
    seed=$seed iter=100000 note=ps/relu/dirichletx256."$seed"  bash run/.sh --pde poisson.dirichlet --model fno --mode 31 31 --grid 256
    seed=$seed iter=100000 note=ps/relu/dirichletx256C."$seed" bash run/.sh --pde poisson.dirichlet --model fno --mode 31 31 --grid 256 --cheb --vmap 16

    seed=$seed iter=100000 note=ps/tanh/dirichlet:NSM."$seed"  bash run/.sh --pde poisson.dirichlet --model fno --mode 31 31 --spectral --activate tanh
    seed=$seed iter=100000 note=ps/tanh/dirichletx64."$seed"   bash run/.sh --pde poisson.dirichlet --model fno --mode 31 31 --grid 64  --activate tanh
    seed=$seed iter=100000 note=ps/tanh/dirichletx128."$seed"  bash run/.sh --pde poisson.dirichlet --model fno --mode 31 31 --grid 128 --activate tanh
    seed=$seed iter=100000 note=ps/tanh/dirichletx256."$seed"  bash run/.sh --pde poisson.dirichlet --model fno --mode 31 31 --grid 256 --activate tanh
    seed=$seed iter=100000 note=ps/tanh/dirichletx256C."$seed" bash run/.sh --pde poisson.dirichlet --model fno --mode 31 31 --grid 256 --activate tanh --cheb --vmap 16

done