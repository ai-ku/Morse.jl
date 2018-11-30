#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=high
#SBATCH --mem-per-cpu=16GB
#SBATCH --cpus-per-task=1
#SBATCH --job-name="trmor2018_0.1_full"
#SBATCH --output=trmor2018_0.1_full.out
#SBATCH --error=trmor2018_0.1_full.error

echo "julia main.jl --dataSet TRDataSet --version 2018 --epochs 100 --trainSize 3467 --lemma --dropouts 0.3 --modelType S2S"
julia main.jl --dataSet TRDataSet --version 2018 --epochs 100 --trainSize 3467 --lemma --dropouts 0.3 --modelType S2S
