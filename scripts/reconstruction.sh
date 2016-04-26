mkdir reconstruction_res

LATENT_SIZES=( '2' '5' '10' '20' )
for LATENT_SIZE in "${LATENT_SIZES[@]}"
do
    ./run_on_gpu.sh --save_file reconstruction_res/VAE_continuous_${LATENT_SIZE}.mdl --continuous --n_latent $LATENT_SIZE --n_epochs 70000
done

LATENT_SIZES=( '3' '5' '10' '20' '200' )
for LATENT_SIZE in "${LATENT_SIZES[@]}"
do
    ./run_on_gpu.sh --save_file reconstruction_res/VAE_discrete_${LATENT_SIZE}.mdl --n_latent $LATENT_SIZE --n_epochs 3000
done
