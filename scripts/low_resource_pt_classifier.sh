#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=high
#SBATCH --mem-per-cpu=16GB
#SBATCH --cpus-per-task=1
#SBATCH --job-name="low_pt_classifer"
#SBATCH --output=low_pt_classifer.out
#SBATCH --error=low_pt_classifer.error

echo "julia main.jl --lang pt --epochs 100 --dropouts 0.3 --modelType Classifier --optimizer 'Rmsprop(lr=2.5e-3)'"
julia main.jl --lang pt --epochs 100 --dropouts 0.5 --modelType Classifier --optimizer 'Rmsprop(lr=1.0e-3, gclip=60)'
