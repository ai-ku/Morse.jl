#!/bin/bash
#SBATCH --partition=ai
#SBATCH --account=ai
#SBATCH --gres=gpu:1
#SBATCH --qos=ai
#SBATCH --time=56:00:00
#SBATCH --mem-per-cpu=32GB
#SBATCH --cpus-per-task=1
#SBATCH --nodelist=dy02
#SBATCH --job-name="trmor2016_full"
#SBATCH --output=trmor2016_full.out
#SBATCH --error=trmor2016_full.error

echo "julia main.jl --dataSet TRDataSet --version 2016 --epochs 100 --lemma --dropouts 0.3"
/scratch/users/eakyurek13/julia-1.0.1/bin/julia main.jl --dataSet TRDataSet --version 2016 --epochs 100 --lemma --dropouts 0.3
