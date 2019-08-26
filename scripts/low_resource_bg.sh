#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=high
#SBATCH --mem-per-cpu=16GB
#SBATCH --cpus-per-task=1
#SBATCH --job-name="low_bg"
#SBATCH --output=low_bg.out
#SBATCH --error=low_bg.error


echo "julia main.jl --lang bg --epochs 100 --dropouts 0.5 --modelType S2SContext"
julia main.jl --lang bg --epochs 100 --dropouts 0.5 --modelType S2SContext
#echo "julia main.jl --lang bg --epochs 100 --lemma --dropouts 0.5 --modelType S2SContext"
#julia main.jl --lang bg --epochs 100 --lemma --dropouts 0.5 --modelType S2SContext
