#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=high
#SBATCH --mem-per-cpu=16GB
#SBATCH --cpus-per-task=1
#SBATCH --job-name="transfer100"
#SBATCH --output=transfer100.out
#SBATCH --error=transfer100.error


julia main.jl --lang sv --epochs 100 --trainSize 1000  \
              --dropouts 0.5 --mode 3\
              --sourceModel ../checkpoints/morse.MorseDis_lemma_true_lang_UD-da_size_full_epoch10.jld2

julia main.jl --lang bg --epochs 100 --trainSize 1000  \
              --dropouts 0.5 --mode 3\
              --sourceModel ../checkpoints/morse.MorseDis_lemma_true_lang_UD-ru_size_full_epoch10.jld2


julia main.jl --lang hu --epochs 100 --trainSize 1000  \
              --dropouts 0.5 --mode 3\
              --sourceModel ../checkpoints/morse.MorseDis_lemma_true_lang_UD-fi_size_full_epoch10.jld2

julia main.jl --lang pt --epochs 100 --trainSize 1000  \
            --dropouts 0.5 --mode 3\
            --sourceModel ../checkpoints/morse.MorseDis_lemma_true_lang_UD-es_size_full_epoch10.jld2
