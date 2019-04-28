#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=high
#SBATCH --mem-per-cpu=16GB
#SBATCH --cpus-per-task=1
#SBATCH --job-name="log_hu_classifer"
#SBATCH --output=low_hu_classifer.out
#SBATCH --error=low_hu_classifer.error

echo "julia main.jl --lang hu --epochs 100 --dropouts 0.3 --modelType Classifier --optimizer 'Rmsprop(lr=2.5e-3)'"
julia main.jl --lang hu --epochs 100 --dropouts 0.5 --modelType Classifier --optimizer 'Rmsprop(lr=1.0e-3, gclip=60)'
