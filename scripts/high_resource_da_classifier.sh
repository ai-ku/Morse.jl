#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH --qos=high
#SBATCH --mem-per-cpu=16GB
#SBATCH --cpus-per-task=1
#SBATCH --job-name="high_da_classifer"
#SBATCH --output=high_da_classifer.out
#SBATCH --error=high_da_classifer.error


echo "julia main.jl --lang da --epochs 100 --dropouts 0.3 --modelType Classifier --optimizer 'Rmsprop(lr=2.5e-3)'"
julia main.jl --lang da --epochs 100 --dropouts 0.3 --modelType Classifier --optimizer 'Rmsprop(lr=2.5e-3)'
