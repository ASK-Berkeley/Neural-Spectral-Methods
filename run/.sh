XLA_PYTHON_CLIENT_PREALLOCATE=True \
XLA_PYTHON_CLIENT_MEM_FRACTION=.95 \
python main.py \
    --seed ${seed:-$RANDOM} \
        --hdim 64 \
        --depth 4 \
        --activate relu \
    $@ train \
        --bs ${bs:-16} \
        --lr ${lr:-1e-3} \
        --schd ${schd:-exp} \
        --iter ${iter:-30000} \
        --vmap ${vmap:-""} \
        --ckpt ${ckpt:-100} \
        --note ${note:-"$(date)"}