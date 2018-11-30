#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=high
#SBATCH --mem-per-cpu=16GB
#SBATCH --cpus-per-task=1
#SBATCH --job-name="high_da"
#SBATCH --output=high_da.out
#SBATCH --error=high_da.error

echo "julia main.jl --lang da --epochs 100 --dropouts 0.3"
julia main.jl --lang da --epochs 100 --dropouts 0.3
echo "julia main.jl --lang da --epochs 100 --lemma --dropouts 0.3"
julia main.jl --lang da --epochs 100 --lemma --dropouts 0.3
