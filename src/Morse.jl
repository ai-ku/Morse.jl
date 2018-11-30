module Morse

using KnetLayers, ArgParse, Random
import KnetLayers: _pack_sequence, Knet.save, Knet.load

dir(path...) = joinpath(dirname(@__DIR__),path...)

include("parser.jl"); export Analysis, EncodedAnalysis, ParsedIO, EncodedIO,
                             IndexedDict, Vocabulary, Parser, parseDataLine,
                             parseFile, encode, specialTokens,
                             UDDataSet, TRDataSet, TRtoLower;

include("minibatch.jl"); export SentenceBatch, miniBatch;

include("code2lang.jl"); export CODE_TO_LANG;

include("models.jl"); export Sequential, Discriminative, Disambiguator,
                             MorseModel, S2S, S2SContext, Classifier, S2SContexDis,
                             WordEncoder,ContextEncoder, OutputEncoder, Decoder,
                             loss, predict, MorseDis, S2SContexDis;

include("util.jl"); export download, loadModel, saveModel, setoptim!, lrdecay!,
                           splitdata, createsplits, prepareData, StringAnalysis,
                           evaluate, F1average, F1update!, getLabels,
                           makeFormat, printFormat;

include("train.jl"); export trainmodel!, trainepoch!;

end # module
