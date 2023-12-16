for re in 4 3 2; do

    note=ns.T3/re"$re":NSM."$seed" iter=200000 ckpt=1000        bash run/.sh --pde navierstokes.re"$re" --model fno --hdim 32 --depth 10 --mode 12 31 31 --spectral
    note=ns.T3/re"$re"x64."$seed"  iter=200000 ckpt=1000 vmap=1 bash run/.sh --pde navierstokes.re"$re" --model fno --hdim 32 --depth 10 --mode 12 31 31 --grid 64
    note=ns.T3/re"$re"x96."$seed"  iter=200000 ckpt=1000 vmap=1 bash run/.sh --pde navierstokes.re"$re" --model fno --hdim 32 --depth 10 --mode 12 31 31 --grid 96

done