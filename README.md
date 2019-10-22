# Morse

Morse is the morphological analysis model described in:

Akyürek, Ekin, Erenay Dayanık, and Deniz Yuret. "Morphological Analysis Using a Sequence Decoder." *Transactions of the Association for Computational Linguistics* 7 (2019): 567-579. ([TACL](https://www.transacl.org/ojs/index.php/tacl/article/view/1654), [arXiv](https://arxiv.org/abs/1805.07946)).

## Dependencies
  - Julia 1.x
  - Network connection

## Installation

```SHELL
   git clone https://github.com/ai-ku/Morse.jl
   cd Morse.jl
```

**Note**: Setup and Data is optional because running an experiment from the scripts directory automatically sets up the environment and installs required data when needed. However, if you're working in a cluster node that has no internet connection, you may need to perform these steps manually.

* #### Setup (Optional)
```JULIA
   (v1.1) pkg> activate .
   (v1.1) Morse> instantiate
```

* #### Data (Optional)
```JULIA
   julia> using Morse
   julia> download(TRDataSet)
   julia> download(UDDataSet)
```

## Experiments

To verify the results presented in the paper, you may run the scripts to train models and ablations. During training logs will be created at [logs/](logs/) folder.

Detailed information about experiments can be found in [scripts/](scripts/README.md)

**Note**: An Nvidia GPU is required to train the models in a reasonable amount of time.
