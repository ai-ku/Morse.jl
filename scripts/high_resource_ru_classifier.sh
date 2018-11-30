#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=high
#SBATCH --mem-per-cpu=16GB
#SBATCH --cpus-per-task=1
#SBATCH --job-name="high_ru_classifer"
#SBATCH --output=high_ru_classifer.out
#SBATCH --error=high_ru_classifer.error


echo "julia main.jl --lang ru --epochs 100 --dropouts 0.3 --modelType Classifier --optimizer 'Rmsprop(lr=2.5e-3)'"
julia main.jl --lang ru --epochs 100 --dropouts 0.3 --modelType Classifier --optimizer 'Rmsprop(lr=2.5e-3)'
