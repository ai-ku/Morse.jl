# Morse

[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://ekinakyurek.github.io/Morse.jl/latest)
[![](https://gitlab.com/JuliaGPU/Morse/badges/master/pipeline.svg)](https://gitlab.com/JuliaGPU/Morse/pipelines)
[![](https://travis-ci.org/ekinakyurek/Morse.jl.svg?branch=master)](https://travis-ci.org/ekinakyurek/Morse.jl)
[![codecov](https://codecov.io/gh/ekinakyurek/Morse.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/ekinakyurek/Morse.jl)

Morse is a collection of morphological taggers presented in the [paper](https://arxiv.org/abs/1805.07946v1) that you can train on your data.

Furthermore, Morse provides pre-trained models which are trained in [Universal Dependencies](http://universaldependencies.org)
and in [TrMor](https://github.com/ai-ku/TrMor2018) datasets, so you can tag your sentences immediately.

## Dependencies
  - Julia 1.1
  - Network connection

## Install

### For User
```JULIA
   (v1.1) pkg> add https://github.com/ai-ku/Morse.jl
```
### For Developer
```JULIA
   (v1.1) pkg> dev https://github.com/ai-ku/Morse.jl
```
### For Exact Replication

```SHELL
   git clone https://github.com/ai-ku/Morse.jl
   cd Morse.jl
```
* #### Setup (Optional)
Note: It is optional because running experiments automatically setup the environment and install required data if needed. However, if you didn't run any experiment and want to work on REPL immediately you need to instantiate and download datasets.
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

Note: Nvidia GPU is required to train on a reasonable time.

## Tagging

Note: coming very soon...

```Julia
   julia> using Morse
   julia> model = download(MorseModel, format=UDDataSet, lang="en")
   julia> model("I have no purpose but to make others' lives easier.")
```

## Customized Training

Note: Nvidia GPU is required to train on a reasonable time.

```Julia
   julia> using Morse
   julia> config = Morse.intro([]) # default configuration but you can modify
   julia> config[:logFile] = nothing # to print stdout.
   julia> dataFiles = ["train.txt", "test.txt"]
   julia> data, vocab, parser = prepareData(dataFiles,TRDataSet) # or UDDataSet
   julia> model = MorseModel(config,vocab)
   julia> trainmodel!(model,data,config,vocab,parser) # can take hours or more depends to your data
```
