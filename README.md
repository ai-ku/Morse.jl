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

## Installation

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
Note: It is optional because running an experiment automatically setups the environment and installs required data (if needed). However, if you didn't run any experiment and want to work on REPL immediately, you need to instantiate and download datasets.
```JULIA
   (v1.1) pkg> activate .
   (v1.1) Morse> instantiate # only in the first time
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

Note: Limited Support

```Julia
   julia> using Knet, KnetLayers, Morse
   julia> model, vocabulary, parser = trained(MorseModel, TRDataSet, vers="2018");
   julia> predictions = model("annem sana yardım edemez .", v=vocabulary, p=parser)
   annem anne+Noun+A3sg+P1sg+Nom
   sana sen+Pron+Pers+A2sg+Pnon+Dat
   yardım yardım+Noun+A3sg+Pnon+Nom
   edemez et+Verb^DB+Verb+Able+Neg+Aor+A3sg
   . .+Punct
```

## Customized Training

Note: Nvidia GPU is required to train on a reasonable time.

```Julia
   julia> using Knet, KnetLayers, Morse
   julia> config = Morse.intro(split("--logFile nothing --lemma --dataSet TRDataSet")) # you can modify the program arguments
   julia> dataFiles = ["train.txt", "test.txt"] # make sure you have theese files exists in the given path
   julia> data, vocab, parser = prepareData(dataFiles,TRDataSet) # or UDDataSet
   julia> data = miniBatch(data,vocab) # sentence minibatching is required for processing a sentence correctly
   julia> model = MorseModel(config,vocab)
   julia> setoptim!(model, SGD(;lr=1.6,gclip=60.0))
   julia> trainmodel!(model,data,config,vocab,parser) # can take hours or more depends to your data
   julia> predictions = model("Annem sana yardım edemez .", v=vocab, p=parser)
```
