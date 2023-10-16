config="ivae.yaml"
run_path="./running_related_data"
documentation="iVAE"
num_simulations=1
seed=0

python ./main.py --config $config \
                 --run $run_path \
                 --doc $documentation \
                 --n-sims $num_simulations \
                 --seed $seed 