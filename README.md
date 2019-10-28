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

**Note**: Setup and Data is optional because running an experiment from the scripts directory automatically sets up the environment and installs required data when needed. However, if you're working in a cluster node that has no internet connection, you may need to perform these steps manually. To get the `pkg>` prompt in Julia for package commands please use the ']' key. Backspace gets back to the original `julia>` prompt.

* #### Setup (Optional)
```JULIA
   julia> # Press the `]` key to get the `pkg>` prompt
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

To verify the results presented in the paper, you may run the scripts to train models and ablations. During training logs will be created at [logs/](logs/) folder.

Detailed information about experiments can be found in [scripts/](scripts/README.md)

**Note**: An Nvidia GPU is required to train the models in a reasonable amount of time.

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
