#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=high
#SBATCH --mem-per-cpu=16GB
#SBATCH --cpus-per-task=1
#SBATCH --job-name="low_bg_classifer"
#SBATCH --output=low_bg_classifer.out
#SBATCH --error=low_bg_classifer.error


echo "julia main.jl --lang bg --epochs 100 --dropouts 0.5 --modelType Classifier --optimizer 'Rmsprop(lr=1.0e-3, gclip=60)'"
julia main.jl --lang bg --epochs 100 --dropouts 0.5 --modelType Classifier --optimizer 'Rmsprop(lr=1.0e-3, gclip=60)'
