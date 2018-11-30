# Reference

**Contents**

```@contents
Pages = ["reference.md"]
```

## Models
```@docs
Morse.Model
Morse.WordEncoder
Morse.ContextEncoder
Morse.OutputEncoder
Morse.Decoder
Morse.Sequential
Morse.Discriminative
Morse.Disambiguator
Morse.MorseModel
Morse.S2S
Morse.S2SContext
Morse.Classifier
Morse.MorseDis
Morse.S2SContexDis
Morse.loss
Morse.predict
Morse.recurrentLoss
Morse.recurrentPredict
```

## Parser

```@docs
Morse.Analysis
Morse.EncodedAnalysis
Morse.ParsedIO
Morse.EncodedIO
Morse.IndexedDict
Morse.Vocabulary
Morse.Parser
Morse.parseDataLine
Morse.parseFile
Morse.encode
Morse.specialTokens
Morse.UDDataSet
Morse.TRDataSet
Morse.TRtoLower
 Morse.DataSet
```

## Minibatch

```@docs
Morse.SentenceBatch
Morse.miniBatch
```

## Util

```@docs
Morse.prepareData
Morse.splitdata
Morse.createsplits
Morse.evaluate
Morse.download
Morse.loadModel
Morse.saveModel
Morse.setoptim!
Morse.lrdecay!
Morse.printLog
Morse.printConfigs
Morse.StringAnalysis
Morse.F1average
Morse.F1update!
Morse.getLabels
Morse.makeFormat
Morse.printFormat
```

## Train
```@docs
Morse.intro
Morse.main
Morse.trainepoch!
```

## Code2Lang
```@docs
Morse.CODE_TO_LANG
```

## Function Index

```@index
Pages = ["reference.md"]
```
