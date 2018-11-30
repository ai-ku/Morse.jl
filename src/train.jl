using Random, AutoGrad, Knet, KnetLayers
"""
    intro(args)

    Create configuration dictionary given program arguments.
"""
function intro(args)
    s = ArgParseSettings()
    @add_arg_table s begin
        ("--dataSet"; arg_type=String; default="UDDataSet"; help="TRDataSet or UDDataSet")
        ("--langcode"; arg_type=String; default="fi"; help="lang code for UDDataSet")
        ("--version"; arg_type=Int; default=2018; help="2006 | 2016 | 2018 for TRDataSet")
        ("--bestModel"; default="../checkpoints/bestModel"; help="path to best model")
        ("--modelFile"; arg_type=String; help="load a pretrained model")
        ("--sourceModel"; arg_type=String)
        ("--trainSize"; arg_type=Int; default=typemax(Int))
        ("--logFile"; arg_type=String; default="../logs/morse"; help="log file")
        ("--genFile"; arg_type=String; default="../generations/morse"; help="file for generations")
        ("--hiddenSizes"; nargs='+';arg_type=Int; default=[512,512,512,512];
            help="char-encoder, context-encoder, decoder-l1, decoder-l2")
        ("--embedSizes"; nargs='+';arg_type=Int; default=[64,256])
            help="char and output embeddings"
        ("--batchSize"; arg_type=Int; default=1; help="batch size for training")
        ("--atype"; default=(gpu()>=0 ? "KnetArray{Float32}" : "Array{Float32}"))
        ("--seed"; arg_type=Int; default=31; help="random number seed.")
        ("--previous"; arg_type=Int; default=2; help="output encoder window")
        ("--dropouts"; arg_type=Float64; default= 0.3; help="dropout probabilities")
        ("--decayRate"; arg_type=Float64; default=0.8; help="learning rate decay")
        ("--optimizer"; arg_type=String; default="SGD(;lr=1.6,gclip=60.0)")
        ("--patiance"; arg_type=Int; default=10; help="number of validation to wait")
        ("--mode"; arg_type=Int;default=1; help="1=training, 2=evaluate, 3=transfer")
        ("--epochs"; arg_type=Int; default=1; help="# of epochs for training.")
        ("--wordLimit"; arg_type=Int; default=100; help="maximum # of words in a sentence.")
        ("--threshold"; arg_type=Float64; default=0.0; help="threshold to be considered")
        ("--modelType"; arg_type=String; default="MorseModel";
            help="MorseModel | S2S | S2SContext | Classifier")
        ("--lemma"; action = :store_true;
            help = "predict lemma+morph feats (only for sequential models)")
    end
    config = parse_args(args,s;as_symbols=true)

    lang   = config[:dataSet]   == "TRDataSet"  ? string("TR-tr",config[:version]) : string("UD-",config[:langcode])
    tsize  = config[:trainSize] == typemax(Int) ? "full"  : string(config[:trainSize],"sent")

    postfix = string(".",config[:modelType],"_lemma_",config[:lemma],"_lang_",lang,"_size_",tsize)
    config[:bestModel] = config[:bestModel]*postfix
    config[:logFile] = config[:logFile]*postfix
    config[:genFile] = config[:genFile]*postfix
    config[:bestTestAcc] = config[:bestDevAcc] = 0.0

    if isfile(config[:bestModel]) && config[:modelFile] === nothing
        config[:modelFile] = config[:bestModel]*".jld2"
    end

    return config
end

"""
    main(ARGS=[]; config = intro(ARGS))

    Main procedure according to config. see `intro`
"""
function main(ARGS=[]; config = intro(ARGS))

    logFile = config[:logFile]!==nothing ? open(config[:logFile]*".log","a+") : nothing

    DataSet  = eval(Meta.parse(config[:dataSet]))
    download(DataSet) # downloads if dataset is not available in the data/

    ModelType = eval(Meta.parse(config[:modelType]))

    if DataSet === UDDataSet
        code = config[:langcode]
        lang = CODE_TO_LANG[code]
        files = Morse.dir("data","ud-treebanks-v2.1/UD_$(lang)") *
                        "/$(code)-ud-" .* ["train","dev","test"] .* ".conllu"
    else # DataSet === TRDataSet
        vers = config[:version]
        splits = vers==2018 ? [".train"] : [".train",".test"]
        files = Morse.dir("data","TrMor2018")*"/TrMor$(vers)/trmor$(vers)".*splits
    end

    data, vocab, parser = prepareData(files, DataSet; vers=config[:version],
                                                      tsize=config[:trainSize],
                                                      seed=config[:seed],
                                                      withLemma=config[:lemma],
                                                      parseAll=(ModelType<:Disambiguator))

    data = miniBatch(data, vocab)

    ArrayType = eval(Meta.parse(config[:atype]))
    KnetLayers.settype!(ArrayType)
    KnetLayers.Knet.seed!(config[:seed])


    printConfig(logFile,config)
    printLog(logFile,"dataLengths (train,dev,test): ",length.(data))
    printLog(logFile,"vocabLengths", (chars=length(vocab.chars),
                                      tags=length(vocab.tags),
                                      comptags=length(vocab.comptags),
                                      words=length(vocab.words)))

    if config[:mode] == 1 # train
        if  config[:modelFile] !== nothing
            model,o,_,_ = loadModel(config[:modelFile])
            config[:bestDevAcc]  = o[:bestDevAcc]
            config[:bestTestAcc] = o[:bestTestAcc]
        else
            model = ModelType(config, vocab)
            setoptim!(model, eval(Meta.parse(config[:optimizer])))
        end

        trainmodel!(model, data, config, vocab, parser; logFile=logFile)

        if isfile(config[:bestModel]*".jld2")
            model,_,_,_ = loadModel(config[:bestModel]*".jld2")
        end
    elseif config[:mode] == 2 # generate
        model,_,_,_ = loadModel(config[:modelFile])
    elseif config[:mode] == 3
        # transfer learning
    end

    #Final Generation for Test
    open(config[:genFile]*".gen","w") do f
        scores = evaluate(model, last(data), vocab, parser, file=f)
        printLog(logFile, "epoch=final | set=Test | scores: ", scores)
    end

    #Final Evaluation for Logging
    for (set,name) in zip(data,(:Train, :Dev))
        scores = evaluate(model, set, vocab, parser, file=nothing)
        printLog(logFile, "epoch=final | set=$name | scores: ", scores)
    end

    return model, vocab, parser, config, data
end

"""
    trainmodel!(M::Model, data::Vector, o::Dict{Symbol,Any}, v::Vocabulary,
                           p::Parser; logFile::IO=stdout, patiance=o[:patiance])

    Trains a `Model` on given data.
    Make sure that you `setoptim!` for model parameters before calling `trainepoch!`

    It applies learning rate decay and early stopping according to patiance argument
    It also does evaluations and logs them after each epoch.
"""

function trainmodel!(M::Model, data::Vector, o::Dict{Symbol,Any}, v::Vocabulary,
                            p::Parser; logFile::IO=stdout, patiance=o[:patiance])
    for i=1:o[:epochs]
        trnloss = trainepoch!(M, data[1], v; wordLimit=o[:wordLimit])

        printLog(logFile, "epoch=$i | set=Train | scores: ", (loss=trnloss[1]/trnloss[2],))
        for (k,set) in enumerate((:Dev,:Test))
            scores = evaluate(M, data[k+1], v, p, file=nothing)
            printLog(logFile, "epoch=$i | set=$set | scores: ", scores)
            acc = scores.accuracies.complete
            if acc > o[Symbol(:best,set,:Acc)]
                o[Symbol(:best,set,:Acc)] = acc;
                if set == :Dev
                    patiance = o[:patiance]
                    saveModel(o[:bestModel]*".jld2", M, o, v, p)
                end
                printLog(logFile,"Current Best $set Accuracies: ", scores.accuracies)
            end
        end

        i%10==0 && saveModel(o[:bestModel]*"_epoch$i.jld2", M, o, v, p)

        if (patiance -= 1) == o[:patiance] ÷ 2
            lrdecay!(M,o[:decayRate])
        elseif patiance < 0
            printLog(logFile,"patiance < 0, training stops")
            break
        end

    end
end

"""
    trainepoch!(M::Model, data::Vector{SentenceBatch}, v::Vocabulary; wordLimit=100)

    Trains a `Model` one epoch on given data.
    Make sure that you `setoptim!` for model parameters before calling `trainepoch!``
"""
function trainepoch!(M::Model, data::Vector{SentenceBatch}, v::Vocabulary; wordLimit=100)
    parameters = params(M)
    trainloss = zeros(2)
    for (i,d) in enumerate(shuffle(data))
        length(d.encodedIOs) > wordLimit && continue
        J = @diff loss(M, d; v=v)
        isnan(value(J)) && continue
        if isa(M,Sequential)
            trainloss .+= [value(J),1] .* sum(d.masks)
        else
            trainloss .+= [value(J),1] .* length(d.encodedIOs)
        end
        for p in parameters
            update!(value(p), grad(J,p), p.opt)
        end
    end
    return trainloss
end
