#!/bin/bash
#SBATCH --partition=ai
#SBATCH --account=ai
#SBATCH --gres=gpu:1
#SBATCH --qos=ai
#SBATCH --time=56:00:00
#SBATCH --mem-per-cpu=32GB
#SBATCH --cpus-per-task=1
#SBATCH --nodelist=dy02
#SBATCH --job-name="trmor2006_full_dis"
#SBATCH --output=trmor2006_full_dis.out
#SBATCH --error=trmor2006_full_dis.error

/scratch/users/eakyurek13/julia-1.0.1/bin/julia main.jl --dataSet TRDataSet --version 2006 --epochs 100 --lemma --dropouts 0.3 --modelType MorseDis
