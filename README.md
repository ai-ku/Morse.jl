# Morse

Morse is a collection of morphological taggers from the [paper](https://arxiv.org/abs/1805.07946v1).

**Note**: This repo is created just for you to validate the results presented in the paper. If you want to use the models in different settings please refer to this [repo](https://github.com/ekinakyurek/Morse.jl).

## Dependencies
  - Julia 1.1
  - Network connection

## Installation

```SHELL
   git clone https://github.com/ai-ku/Morse.jl
   cd Morse.jl
```
**Note**: Setup and Data is optional because running an experiment automatically setups the environment and installs required data when needed. However, if you're working in a cluster nodes that has no internet connection, you should do these steps, before running the experiments , in the login node that has a connection.

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

To verify the results presented in the paper, you may run the scripts to train models an ablations. During training logs will be created at [logs/](logs/) folder.

Detailed information about experiments can be found in [scripts/](scripts/README.md)

**Note**: Nvidia GPU is required to train on a reasonable time.