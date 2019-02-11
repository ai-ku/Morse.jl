using Dates, LibGit2
import Base: download, show

function Base.show(io::IO, x::Union{Float64,Float32})
    Base.Grisu._show(io, round(x, sigdigits=4), Base.Grisu.SHORTEST, 0,
                     get(io, :typeinfo, Any) !== typeof(x), false)
end

Counter()  = zeros(Int,2)
Accuracy() = (lemma=Counter(), tag=Counter(), complete=Counter())
F1Scores() = (score=Dict{String,Float64}(), total=Dict{String,Float64}())
F1Metric() = (precision=F1Scores(), recall=F1Scores())

"""
    printLog(f::IO, str...)

    Logger for Morse. Prints current time in eachline and flush the stream
"""
printLog(f::IO, str...) = (println(f,Dates.Time(now()),": ",str...); flush(f);)

"""
    printConfig(f::IO,o::Dict{Symbol,Any})

    Prints the configuration dictionary
"""
function printConfig(f::IO,o::Dict{Symbol,Any})
    printLog(f,"Configuration: ")
    for (k,v) in o; v!== nothing && println(f, k, " => " , v); end
    flush(f)
end

"""
    download(dataset::Type{<:DataSet})

    downloads `DataSet` files.
"""
function download(dataset::Type{UDDataSet})
    if !isdir(dir("data","ud-treebanks-v2.1"))
        download("https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-2515/ud-treebanks-v2.1.tgz",dir("data","ud-treebanks-v2.1.tgz"))
        run(`tar -xf $(dir("data","ud-treebanks-v2.1.tgz")) -C $(dir("data"))`)
        rm(dir("data","ud-treebanks-v2.1.tgz"))
    end
end

function download(dataset::Type{TRDataSet}; path=dir("data","TrMor2018"))
    if !isdir(path)
        repo = LibGit2.clone("https://github.com/ai-ku/TrMor2018", path)
        LibGit2.checkout!(repo,"a0675804beea74e45e9faeac7132070655fca828")
        run(`ln -s  $(joinpath(path,"TrMor2006","trmor2006.train")) $(joinpath(path,"TrMor2016","trmor2016.train"))`)
    end
end

const server_url ="ai.ku.edu.tr/models/morse/"

function download(model::Type{T}, format::Type{TRDataSet}; vers="2018", lemma=true, lang="tr") where T
    flang = format===TRDataSet ? string("TR-tr",vers) : string("UD-",lang)
    mname = string(T,"_lemma_",lemma,"_lang_",flang,"_size_full",".jld2")
    lpath = dir("checkpoints",mname)
    if !isfile(lpath)
        mpath = string(server_url, mname)
        download(mpath,lpath)
    end
    lpath
end

function trained(x...;o...)
    lpath = download(x...;o...)
    model,_,vocabulary,parser = loadModel(lpath)
    return model,vocabulary,parser
end

"""
    loadModel(fname::AbstractString)

    Loads from saved model file. Return tuple of (model,config,vocab,parser)
"""
function loadModel(fname::AbstractString)
    f = KnetLayers.load(fname)
    return f["model"], f["opts"], f["vocab"], f["parser"]
end

"""
    saveModel(fname::AbstractString, model, config, vocab, parser)

    Saves model, config, vocab, parser to a file.
"""
saveModel(fname::AbstractString, model::Model, config::Dict, v::Vocabulary, p::Parser) =
    KnetLayers.save(fname, "model", model, "opts", config, "vocab", v, "parser", p)

"""
   setoptim!(M::Model, optimizer)

   Set optimizer for model parameters
"""
setoptim!(M::Model, optimizer) =
    for p in params(M); p.opt = deepcopy(optimizer); end

"""
   lrdecay!(M::Model, decay::Real)

   Decays learning rate
"""
lrdecay!(M::Model, decay::Real) =
    for p in params(M); p.opt.lr = p.opt.lr*decay; end

"""
   splitdata(data::Vector, r::Vector{<:Real})

   Splits an array according to r.
   r can be Integer array in which every element specifies the exact size of the splits
   r can be Float array in which every element specifies the ratio of the splits
"""
function splitdata(data::Vector, r::Vector{<:Real})
    ls = isa(r[1],AbstractFloat) ? floor.(Int, r .* length(data)) : r #Ratio or Number
    map(rng->data[rng[1]:rng[2]], zip(cumsum(ls) .- ls .+ 1, cumsum(ls)))
end

"""
    createsplits(data::Vector, seed::Integer, tsize::Integer)

    If there is 1 or 2 splits makes it 3 (trn,dev,test)
    according to `seed` and `tsize` inputs.
"""
function createsplits(data::Vector,seed::Integer,tsize::Integer)
    l  = length(data)
    tl = length(data[1])
    shuffle!(MersenneTwister(seed),data[1])
    if l==1
        if tsize < tl
            testsize = min((tl-tsize) ÷ 2, floor(Int,tl*0.1))
            data = splitdata(data[1],[tsize,testsize,testsize])
        else
            data = splitdata(data[1],[0.8,0.1,0.1])
        end
    elseif l==2
        if tsize < tl
            testsize = min(tl-tsize, floor(Int,tl*0.1), 2.5length(data[2]))
            data = push!(splitdata(data[1],[tsize,testsize]),data[2])
        else
            devratio = min(floor(Int,length(data[1])*0.1), 2.5length(data[2])) / length(data[1])
            data = push!(splitdata(data[1],[0.9,devratio]),data[2])
        end
    else
        if tsize < tl
            data[1] = data[1][1:tsize]
        end
    end
    return data
end

"""
   prepareData(files::Vector{<:AbstractString}, dtype::Type{<:DataSet};
                           seed=31, tsize=typemax(Int), withLemma=true)

   preparesData for training and evaluation for datafiles.
   returns encodedData, vocabulary, parser
"""
function prepareData(files::Vector{<:AbstractString}, dtype::Type{<:DataSet};
                     vers=2018, seed=31, tsize=typemax(Int), withLemma=true, parseAll=false)
    parser = Parser{dtype}(vers)
    data = createsplits(parseFile.(files; p=parser, withLemma=withLemma, parseAll=parseAll), seed, tsize)
    vocab = Vocabulary(data)
    return encode.(data, v=vocab), vocab, parser
end

"""
    StringAnalysis

    Keeps analysis of a word with String fields.
"""
struct StringAnalysis
    word::String
    lemma::String
    tags::Vector{String}
end

# Sequential
function StringAnalysis(word::Vector{Int}, seq::Vector{Int}; v::Vocabulary, isgold=false)
    lemma, tags = String[], String[]
    for index in seq
        if index == v.specialIndices.eow
            push!(tags,v.specialTokens.eow); break
        end
        !isgold && index ∈ v.specialIndices && continue
        str = v.tags[index]
        length(str)==1 ? push!(lemma,str) : push!(tags,str)
    end
    StringAnalysis(join(v.chars[word]), join(lemma), tags)
end

# Discriminative
function StringAnalysis(word::Vector{Int}, output::Int; v::Vocabulary, isgold=false)
    StringAnalysis(join(v.chars[word]), "",
                   push!(split(v.comptags[output],'|'),v.specialTokens.eow))
end

# Vocabulary is needed for digit masking
function getLabels(pred::StringAnalysis, gold::StringAnalysis; v::Vocabulary)
    if length(gold.lemma) != length(pred.lemma)
        return (false, pred.tags == gold.tags)
    end
    for (gc,pc) in zip(gold.lemma, pred.lemma)
        isdigit(gc) && continue
        gc != pc && return (false,pred.tags == gold.tags)
    end
    return (true, pred.tags == gold.tags)
end

function makeFormat(a::StringAnalysis, p::Parser{UDDataSet})
    lemma = length(a.lemma) == 0 ? "X" : join(a.lemma)
    tags = a.tags[1:end-1]
    posTag = length(tags) == 0 ? "X" : tags[1]
    morphFeats = length(tags) < 2 ? p.unkToken : join(tags[2:end],p.tagsSeperator)
    return (a.word, lemma, posTag, morphFeats)
end

printFormat(predS,parser; f=stdout) =
    for (i,pred) in enumerate(predS) printFormat(f,i,pred,parser) end

function printFormat(f::IO, i::Integer, predStr::NTuple, p::Parser{UDDataSet})
    word, lemma, pos_tag, morph_feats = predStr
    s,u = p.partsSeperator, p.unkToken
    write(f,string(i),s,word,s,lemma,s,pos_tag,s,u,s,morph_feats,s,u,s,u,s,u,s,u,'\n')
end

function makeFormat(a::StringAnalysis, p::Parser{TRDataSet})
    lemma  = length(a.lemma) == 0 ? "" : join(a.lemma)
    tags   = a.tags[1:end-1]
    morphFeats = join(tags,p.tagsSeperator)
    return (a.word, lemma, "", morphFeats)
end

function printFormat(f::IO, i::Integer, predStr::NTuple, p::Parser{TRDataSet})
    word, lemma, pos_tag, morph_feats = predStr
    write(f,word,p.partsSeperator,lemma,p.tagsSeperator,morph_feats,'\n')
end

function evaluate(M::Model, data::Vector{SentenceBatch}, v::Vocabulary, p::Parser{T}; file=nothing) where T
    # Initialize metrics, amb: accuracy metric in ambigious words
    loss, acc, amb, F1 = zeros(2), Accuracy(), Accuracy(), F1Metric()

    for d in data
        # Forward Run
        preds, J = predict(M, d; v=v)
        # Cross Entropy Loss Metric
        if isa(M,Sequential)
            loss .+= [J, sum(d.masks)]
        else
            loss .+= [J, 1] .* length(d.encodedIOs)
        end

        if file !== nothing
            write(file, p.sentenceStart, " = ",
                  join(map(a->join(v.chars[a.chars]), d.encodedIOs)," "),
                  '\n')
        end

        @inbounds for (i,encodedIO) in enumerate(d.encodedIOs)

            analysis = encodedIO.analyses[1]  #correct analysis

            if isa(M,Sequential)
                pred  = StringAnalysis(encodedIO.chars, preds[i,:], v=v)
                gold  = StringAnalysis(encodedIO.chars, d.seqOutputs[i,:], v=v, isgold=true)
            else
                pred  = StringAnalysis(encodedIO.chars, preds[i], v=v)
                gold  = StringAnalysis(encodedIO.chars, d.compositeOutputs[i], v=v, isgold=true)
            end

            tf = getLabels(gold, pred, v=v) #(lemma::Bool, tag::Bool, complete::Bool)

            if analysis.isValid

                F1update!(pred.tags, gold.tags, F1.precision, F1.recall, p=p)

                for (accuracy, mul) in zip((acc,amb), (true,encodedIO.isAmbigous))
                    accuracy.lemma     .+= [Int(tf[1]), 1] .* mul
                    accuracy.tag       .+= [Int(tf[2]), 1] .* mul
                    accuracy.complete  .+= [Int(tf[1]&tf[2]), 1] .* mul
                end

            end
            #Generation
            file !== nothing && printFormat(file, i, makeFormat(pred, p), p)
       end
   end
   f1macro, f1micro = F1average(F1.precision, F1.recall)
   return (loss=loss[1]/loss[2], accuracies=percentage(acc), amb=percentage(amb),
            f1macro=f1macro, f1micro=f1micro)
end

percentage(a::NamedTuple) = (lemma=100a.lemma[1]/a.lemma[2], tag=100a.tag[1]/a.tag[2],
                             complete=100a.complete[1]/a.complete[2])

function F1average(precision::NamedTuple, recall::NamedTuple)
    micro_precision = sum(values(precision.score))/sum(values(precision.total))
    micro_recall = sum(values(recall.score))/sum(values(recall.total))
    micro_score = 2(micro_precision * micro_recall)/(micro_precision + micro_recall)

    for k in keys(precision.score)
        precision.score[k] = precision.score[k]/precision.total[k]
    end

    f1_scores = Dict{String,Float64}()
    f1_average = 0.0
    for k in keys(recall.score)
        recall.score[k] = recall.score[k]/recall.total[k]
        if recall.score[k]==0 || !haskey(precision.score,k)
            f1_scores[k] = 0
        else
            f1_scores[k] = 2(precision.score[k] * recall.score[k])/
                             (precision.score[k] + recall.score[k])
        end
        f1_average += recall.total[k] * f1_scores[k]
    end
    f1_average /= sum(values(recall.total))

    return 100(f1_average), 100(micro_score)
end

function F1update!(pred::Vector{String}, gold::Vector{String},
                   precision::NamedTuple, recall::NamedTuple; p::Parser{T}) where T
    if T === UDDataSet
        p_dict = Dict(filter(x->length(x)>1, split.(pred,p.tagValueSeperator)))
        g_dict = Dict(filter(x->length(x)>1, split.(gold,p.tagValueSeperator)))
    else # TRDataSet
        p_dict = Dict(map(x->(x,true),pred))
        g_dict = Dict(map(x->(x,true),gold))
    end
    for (x,y,score,total) in ( (p_dict, g_dict, precision.score, precision.total),
                               (g_dict, p_dict, recall.score, recall.total) )
        for (k,v) in x
            if !haskey(score,k)
                score[k] = total[k] = 0
            end
            if haskey(y,k) && y[k] == v
                score[k] += 1
            end
            total[k] += 1
        end
    end
end

#
# function generate(M::Model, data; v::Vocabulary, p::Parser, prefix="")
#     open(prefix*".generation","w") do f
#         for (i,d) in enumerate(data)
#             preds  = predict(M,d; v=v)
#             write(f, p.sentenceStart, " = ", join(words," "), '\n')
#             for (k,encodedIO) in d.encodedIOs
#                 a = encodedIO.analyses[1] # correct analysis
#                 formatted = makeFormat(StringAnalysis(a.chars, preds[k,:]; v=v), p)
#                 printFormat(f, k, formatted, p)
#             end
#             i != length(data) && write(f,'\n')
#         end
#     end
# end
