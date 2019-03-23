#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=high
#SBATCH --mem-per-cpu=16GB
#SBATCH --cpus-per-task=1
#SBATCH --job-name="low_sv"
#SBATCH --output=low_sv.out
#SBATCH --error=low_sv.error

echo "julia main.jl --lang da --epochs 100 --dropouts 0.3 --modelType S2SContext"
julia main.jl --lang sv --epochs 100 --dropouts 0.3 --modelType S2SContext
echo "julia main.jl --lang da --epochs 100 --lemma --dropouts 0.3 --modelType S2SContext"
julia main.jl --lang sv --epochs 100 --lemma --dropouts 0.3 --modelType S2SContext
