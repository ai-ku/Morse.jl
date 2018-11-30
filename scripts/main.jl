# Downloads Requirements
using Pkg; Pkg.activate("../"); Pkg.instantiate()

using AutoGrad, Knet, KnetLayers, Morse

Morse.main(ARGS)
