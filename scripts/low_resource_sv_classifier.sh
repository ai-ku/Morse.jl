#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH --qos=high
#SBATCH --mem-per-cpu=16GB
#SBATCH --cpus-per-task=1
#SBATCH --job-name="low_sv_classifer"
#SBATCH --output=low_sv_classifer.out
#SBATCH --error=low_sv_classifer.error


echo "julia main.jl --lang sv --epochs 100 --dropouts 0.5 --modelType Classifier --optimizer 'Rmsprop(lr=1.0e-3, gclip=60)'"
julia main.jl --lang sv --epochs 100 --dropouts 0.5 --modelType Classifier --optimizer 'Rmsprop(lr=1.0e-3, gclip=60)'
