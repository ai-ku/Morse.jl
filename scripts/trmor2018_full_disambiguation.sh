#!/bin/bash
#SBATCH --partition=ai
#SBATCH --account=ai
#SBATCH --gres=gpu:1
#SBATCH --qos=ai
#SBATCH --time=56:00:00
#SBATCH --mem-per-cpu=32GB
#SBATCH --cpus-per-task=1
#SBATCH --job-name="trmor2006_full_dis"
#SBATCH --output=trmor2006_full_dis.out
#SBATCH --error=trmor2006_full_dis.error

echo "julia main.jl --dataSet TRDataSet --version 2018 --epochs 100 --lemma --dropouts 0.3 --modelType MorseDis"
julia main.jl --dataSet TRDataSet --version 2018 --epochs 100 --lemma --dropouts 0.3 --modelType MorseDis
