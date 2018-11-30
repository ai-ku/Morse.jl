#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=high
#SBATCH --mem-per-cpu=16GB
#SBATCH --cpus-per-task=1
#SBATCH --job-name="morse_low_1000_tag"
#SBATCH --output=morse_low_1000_tag.out
#SBATCH --error=morse_low_1000_tag.error

for langcode in sv bg hu pt
do
    echo "julia main.jl --lang $langcode --epochs 100 --trainSize 1000 --dropouts 0.5"
    julia main.jl --lang $langcode --epochs 100 --trainSize 1000 --dropouts 0.5
done
