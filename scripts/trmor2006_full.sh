#!/bin/bash
#SBATCH --partition=ai
#SBATCH --account=ai
#SBATCH --gres=gpu:1
#SBATCH --qos=ai
#SBATCH --time=56:00:00
#SBATCH --mem-per-cpu=32GB
#SBATCH --cpus-per-task=1
#SBATCH --job-name="trmor2006_full"
#SBATCH --output=trmor2006_full.out
#SBATCH --error=trmor2006_full.error

echo "julia main.jl --dataSet TRDataSet --version 2006 --epochs 100 --lemma --dropouts 0.3 --patience 6"
julia main.jl --dataSet TRDataSet --version 2006 --epochs 100 --lemma --dropouts 0.3 --patience 6
