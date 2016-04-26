#!/bin/bash

#MNIST
NUM_MNIST_EPOCHS=3000   #enough to get more than 1e8 training samples with each epoch of 50000 examples
LATENT_SIZES=( '3' '5' '10' '20' '200' )
for LATENT_SIZE in "${LATENT_SIZES[@]}"
do
    ./run_on_gpu.sh --n_latent $LATENT_SIZE --n_epochs $NUM_MNIST_EPOCHS --trace_file res/mnist_${LATENT_SIZE}.csv
done

#FREYFACE
NUM_FREY_EPOCHS=70000   #enough to get more than 1e8 training samples with each epoch of 1500 examples
LATENT_SIZES=( '2' '5' '10' '20' )
for LATENT_SIZE in "${LATENT_SIZES[@]}"
do
    ./run_on_gpu.sh --continuous --n_latent $LATENT_SIZE --n_epochs $NUM_FREY_EPOCHS --trace_file res/frey_${LATENT_SIZE}.csv
done
