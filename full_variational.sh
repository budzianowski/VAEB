mkdir full_vb_res
./run_on_gpu.sh --trace_file ./full_vb_res/continuous_2.trc    --vb_param_file ./reconstruction_res/VAE_continuous_2.mdl --full_varational \
    --n_latent 2 --save_file  ./full_vb_res/continuous_2.mdl    --continuous --n_epochs 1000
#./run_on_gpu.sh --trace_file ./full_vb_res/discrete_2.trc      --full_varational --n_latent 2 --save_file  ./full_vb_res/discrete_2.mdl      --n_epochs 100
#./run_on_gpu.sh --trace_file ./full_vb_res/test_discrete.trc   --full_varational --n_latent 10 --save_file ./full_vb_res/test_discrete.mdl   --n_epochs 10
#./run_on_gpu.sh --trace_file ./full_vb_res/test_continuous.trc --full_varational --n_latent 10 --save_file ./full_vb_res/test_continuous.mdl --continuous --n_epochs 10
#./run_on_gpu.sh --trace_file ./full_vb_res/discrete_20.trc     --full_varational --n_latent 20 --save_file ./full_vb_res/discrete_20.mdl     --n_epochs 10
#./run_on_gpu.sh --trace_file ./full_vb_res/continuous_20.trc   --full_varational --n_latent 20 --save_file ./full_vb_res/continuous_20.mdl   --continuous --n_epochs 10
