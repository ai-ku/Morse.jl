#####
##### Abstract Type Definitions
#####

"""
    Model

    Abstract type for all the models in Morse Package
    see `subtypes(Model)`
"""
abstract type Model;end;

"""
    Sequential <: Model

    Abstract type for all sequential models in Morse
    see `subtypes(Sequential)`
"""
abstract type Sequential <: Model;end;

"""
    Discriminative <: Model

    Abstract type for all discriminative models in Morse
    see `subtypes(Discriminative)`
"""
abstract type Discriminative <: Model;end;


#####
##### Primitives
#####

"""
    WordEncoder(config::Dict,v::Vocabulary)
    (M::WordEncoder)(input::Vector{Int}, batchSizes::Vector{Int},
                     unsortingIndices::Vector{Int}; training=false)

    The `WordEncoder` described in the paper.
"""
struct WordEncoder
    embed::Embed
    dropout::Dropout
    encoder::LSTM
end

WordEncoder(o::Dict,v::Vocabulary) = WordEncoder(
    Embed(input=length(v.chars), output=o[:embedSizes][1]),
    Dropout(p=o[:dropouts]),
    LSTM(input=o[:embedSizes][1], hidden=o[:hiddenSizes][1], seed=o[:seed]) |> fgbias!
)

function (M::WordEncoder)(inputs::Vector{Int}, sizes::Vector{Int}, indices::Vector{Int}; training=false)
    out = M.encoder(M.dropout(M.embed(inputs), enable=training), batchSizes=sizes, hy=true, cy=true)
    return out.hidden[:,:,end][:,indices], out.memory[:,:,end][:,indices]
end

"""
    ContextEncoder(config::Dict,v::Vocabulary)
    (M::ContextEncoder)(hiddens; training=false)

    The `ContextEncoder` described in the paper.
"""
struct ContextEncoder
    encoder::LSTM
    reducer::Dense
    dropout::Dropout
end

ContextEncoder(o::Dict,v::Vocabulary) = ContextEncoder(
    LSTM(input=o[:hiddenSizes][1], hidden=o[:hiddenSizes][2], bidirectional=true, seed=o[:seed]) |> fgbias!,
    Dense(input=2o[:hiddenSizes][2],output=o[:hiddenSizes][3], activation=ReLU()),
    Dropout(p=o[:dropouts])
)

function (M::ContextEncoder)(hiddens; training=false)
    input = reshape(hiddens,size(hiddens,1),1,size(hiddens,2))
    y = M.encoder(M.dropout(input; enable=training)).y
    h = M.reducer(reshape(y,size(y,1),size(y,3)))
    return h, zero(value(h)) #TO-DO: zeros
end

"""
    OutputEncoder(config::Dict,v::Vocabulary)
    (M::OutputEncoder)(tagEmbeddings::Vector)
    (M::OutputEncoder)(emb, h, c)

    The `OutputEncoder` described in the paper.
"""
struct OutputEncoder
    encoder::LSTM
    dropout::Dropout
    previous::Int
    hidden
end

OutputEncoder(o::Dict,v::Vocabulary) = OutputEncoder(
    LSTM(input=o[:embedSizes][2], hidden=o[:hiddenSizes][4], seed=o[:seed]) |> fgbias!,
    Dropout(p=o[:dropouts]),
    o[:previous],
    KnetLayers.arrtype(zeros(o[:hiddenSizes][4],1,1))
)

function (M::OutputEncoder)(emb, h, c)
    y = M.encoder(emb, h, c; hy=true, cy=true)
    return y.hidden, y.memory
end

function (M::OutputEncoder)(tagEmbeddings::Vector)
    hos, cos = [], [];
    H,_,_ = size(M.hidden) # assume one layer output encoder
    for i=1:length(tagEmbeddings)
        ho, co = zero(M.hidden), zero(M.hidden) #TO-DO zeros
        for j = max(i-M.previous,1):i-1
             out = M.encoder(M.dropout(tagEmbeddings[j],enable=true), ho, co; hy=true, cy=true)
             ho, co = out.hidden, out.memory
        end
        push!(hos,reshape(ho,H,1))
        push!(cos,reshape(co,H,1))
    end
    return hcat(hos...), hcat(cos...)
end

"""
    Decoder(config::Dict,v::Vocabulary)
    (M::Decoder)(input, hiddens1, hiddens2; training=false)

    The two layer `Decoder` described in the paper.
"""
struct Decoder
    L1::LSTM
    L2::LSTM
    dropout::Dropout
end

function (M::Decoder)(input, hiddens1, hiddens2; training=false)
    out1   = M.L1(M.dropout(input; enable=training), hiddens1...)
    out2   = M.L2(M.dropout(out1.y; enable=training), hiddens2...)
    return (out1.hidden, out1.memory),
           (M.dropout(out2.hidden; enable=training), out2.memory)
end

Decoder(o::Dict,v::Vocabulary) = Decoder(
    LSTM(input=o[:embedSizes][2], hidden=o[:hiddenSizes][3], seed=o[:seed]) |> fgbias!,
    LSTM(input=o[:hiddenSizes][3], hidden=o[:hiddenSizes][3], seed=o[:seed]) |> fgbias!,
    Dropout(p=o[:dropouts])
)

"""
    recurrentLoss(M::Sequential, d::SentenceBatch, hiddens1, hiddens2; startIndex=4)

    Calculate recurrent negative likelihood for sequential models
"""
function recurrentLoss(M::Sequential, d::SentenceBatch, hiddens1, hiddens2; startIndex=4)
    total = 0.0
    input = fill(startIndex, size(d.seqOutputs,1))
    for i=1:size(d.seqOutputs,2)
        hiddens1, hiddens2 = M.decoder(M.outputEmbed(input), hiddens1, hiddens2; training=true)
        h2 = hiddens2[1]
        scores = M.output(reshape(h2,size(h2,1),size(h2,2)))
        # loss calculation
        total += M.loss(scores, d.seqOutputs[:,i] .* d.masks[:,i]; average=false)
        input  = d.seqOutputs[:,i]
    end
    return total
end

"""
    recurrentPredict(M::Sequential, d::SentenceBatch, hiddens1, hiddens2; startIndex=4)

    Make prediciton with recurrent decoder for sequential models except MorseModel due to its `OutputEncoder`
"""
function recurrentPredict(M::Sequential, d::SentenceBatch, hiddens1, hiddens2; startIndex=4)
    wordNumber, timeSteps = size(d.seqOutputs)
    preds = zeros(Int, wordNumber, timeSteps)
    input = fill(startIndex, wordNumber)
    total = 0.0
    for i=1:timeSteps
        hiddens1, hiddens2 = M.decoder(M.outputEmbed(input), hiddens1, hiddens2; training=false)
        h2 = hiddens2[1]
        scores = M.output(reshape(h2,size(h2,1),size(h2,2)))
        # prediction
        input = vec(getindex.(argmax(convert(Array,scores),dims=1),1))
        preds[:,i] = input
        #loss
        total += M.loss(scores, d.seqOutputs[:,i] .* d.masks[:,i]; average=false)
    end
    return preds, total
end

#####
##### Sequential Models
#####

"""
    MorseModel <: Sequential
    MorseModel(config::Dict,v::Vocabulary)
    loss(M::MorseModel, d::SentenceBatch; v::Vocabulary)
    predict(M::MorseModel, d::SentenceBatch; v::Vocabulary)

    The `MorseModel` described in the paper
"""
struct MorseModel <: Sequential
    wordEncoder::WordEncoder
    contextEncoder::ContextEncoder
    outputEncoder::OutputEncoder
    outputEmbed::Embed
    decoder::Decoder
    output::Linear
    loss::CrossEntropyLoss
end

MorseModel(o::Dict,v::Vocabulary) = MorseModel(
    WordEncoder(o,v), ContextEncoder(o,v), OutputEncoder(o,v),
    Embed(input=length(v.tags), output=o[:embedSizes][2]),
    Decoder(o,v),
    Linear(input=o[:hiddenSizes][3], output=length(v.tags)),
    CrossEntropyLoss(dims=1)
)

function loss(M::MorseModel, d::SentenceBatch; v::Vocabulary)
    # Word Encoder
    hiddens2 = M.wordEncoder(d.seqInputs, d.seqSizes, d.unsortingIndices; training=true)
    # Context Encoder
    hiddens1 = M.contextEncoder(hiddens2[1]; training=true)
    # output encoder
    esize = size(M.outputEmbed.weight,1);
    tagEmbeddings = [reshape(M.outputEmbed(d.seqOutputs[i,range]),esize,1,length(range)) for (i,range)=enumerate(d.tagRange)]
    hiddens3 = M.outputEncoder(tagEmbeddings)
    # Decoder
    hiddens2 = hiddens2 .+ hiddens3
    recurrentLoss(M, d, hiddens1, hiddens2; startIndex=v.specialIndices.bow)/sum(d.masks)
end

function predict(M::MorseModel, d::SentenceBatch; v::Vocabulary)
    # Word Encoder
    hiddens2 = M.wordEncoder(d.seqInputs, d.seqSizes, d.unsortingIndices; training=false)
    # Context Encoder
    hiddens1 = M.contextEncoder(hiddens2[1]; training=false)
    # Decoder
    total = 0.0
    wordNumber, timeSteps = size(d.seqOutputs)
    preds = zeros(Int, wordNumber, timeSteps)
    hiddens3  = (zero(M.outputEncoder.hidden), zero(M.outputEncoder.hidden))

    outEmbeddings = [];
    for w=1:wordNumber
        dh2 = (hiddens2[1][:,w:w] .+ hiddens3[1], hiddens2[2][:,w:w] .+ hiddens3[2])
        dh1 = (hiddens1[1][:,w:w], hiddens1[2][:,w:w])

        input = v.specialIndices.bow
        outEmbedding = [];
        continueStoring = true

        for t=1:timeSteps
            rnninput = M.outputEmbed([input])
            if continueStoring && input == v.specialIndices.eow
                continueStoring = false
            elseif continueStoring && input ∉ v.specialIndices && length(v.tags[input]) > 1
                push!(outEmbedding,rnninput)
            end
            dh1, dh2 = M.decoder(rnninput, dh1, dh2; training=false)
            h2 = dh2[1]
            scores = M.output(reshape(h2, size(h2,1), size(h2,2)))
            #prediction
            input = argmax(convert(Array,scores))[1]
            preds[w,t] = input
            # loss calculation
            total += M.loss(scores, d.seqOutputs[w:w,t] .* d.masks[w:w,t]; average=false)
        end

        if length(outEmbedding) != 0

            embedding = hcat(outEmbedding...)
            push!(outEmbeddings, reshape(embedding, size(embedding,1), 1, size(embedding,2)))

            if length(outEmbeddings) > M.outputEncoder.previous
                popfirst!(outEmbeddings);
            end

            if length(outEmbeddings) > 0
                hiddens3  = (zero(M.outputEncoder.hidden), zero(M.outputEncoder.hidden))
                for embs in outEmbeddings
                    hiddens3 = M.outputEncoder(embs, hiddens3[1], hiddens3[2])
                end
            end
        end
    end
    return preds, total
end

"""
    S2S <: Sequential
    S2S(o::Dict,v::Vocabulary)
    loss(M::S2S, d::SentenceBatch; v::Vocabulary)
    predict(M::S2S, d::SentenceBatch; v::Vocabulary)

    Ablation of `MorseModel` where `ContextEncoder` and `OutputEncoder` removed.
"""
struct S2S <: Sequential
    wordEncoder::WordEncoder
    outputEmbed::Embed
    decoder::Decoder
    output::Linear
    loss::CrossEntropyLoss
end

S2S(o::Dict,v::Vocabulary) = S2S(
    WordEncoder(o,v),
    Embed(input=length(v.tags), output=o[:embedSizes][2]),
    Decoder(o,v),
    Linear(input=o[:hiddenSizes][3], output=length(v.tags)),
    CrossEntropyLoss(dims=1)
)

function loss(M::S2S, d::SentenceBatch; v::Vocabulary)
    hiddens2 = M.wordEncoder(d.seqInputs, d.seqSizes, d.unsortingIndices; training=true)
    recurrentLoss(M, d, (nothing, nothing), hiddens2; startIndex=v.specialIndices.bow) / sum(d.masks)
end

function predict(M::S2S, d::SentenceBatch; v::Vocabulary)
    hiddens2 = M.wordEncoder(d.seqInputs, d.seqSizes, d.unsortingIndices; training=false)
    recurrentPredict(M, d, (nothing, nothing), hiddens2; startIndex=v.specialIndices.bow)
end

"""
    S2SContext <: Sequential
    S2SContext(config::Dict,v::Vocabulary)
    loss(M::S2SContext, d::SentenceBatch; v::Vocabulary)
    predict(M::S2SContext, d::SentenceBatch; v::Vocabulary)

    Ablation of the `MorseModel` where `OutputEncoder` removed
"""
struct S2SContext <: Sequential
    wordEncoder::WordEncoder
    contextEncoder::ContextEncoder
    outputEmbed::Embed
    decoder::Decoder
    output::Linear
    loss::CrossEntropyLoss
end

S2SContext(o::Dict,v::Vocabulary) = S2SContext(
    WordEncoder(o,v),
    ContextEncoder(o,v),
    Embed(input=length(v.tags), output=o[:embedSizes][2]),
    Decoder(o,v),
    Linear(input=o[:hiddenSizes][3], output=length(v.tags)),
    CrossEntropyLoss(dims=1)
)

function loss(M::S2SContext, d::SentenceBatch; v::Vocabulary)
    hiddens2 = M.wordEncoder(d.seqInputs, d.seqSizes, d.unsortingIndices; training=true)
    hiddens1 = M.contextEncoder(hiddens2[1]; training=true)
    recurrentLoss(M, d, hiddens1, hiddens2; startIndex=v.specialIndices.bow) / sum(d.masks)
end

function predict(M::S2SContext, d::SentenceBatch; v::Vocabulary)
    hiddens2 = M.wordEncoder(d.seqInputs, d.seqSizes, d.unsortingIndices; training=false)
    hiddens1 = M.contextEncoder(hiddens2[1]; training=true)
    recurrentPredict(M, d, hiddens1, hiddens2; startIndex=v.specialIndices.bow)
end

#####
##### Discriminative Models
#####

"""
    Classifier <: Discriminative
    Classifier(config::Dict,v::Vocabulary)
    loss(M::Classifier, d::SentenceBatch; v::Vocabulary)
    predict(M::Classifier, d::SentenceBatch; v::Vocabulary)

    Ablation of MorseModel where `OutputEncoder` removed and `Decoder` replaced with a classifer.
"""
struct Classifier <: Discriminative
    wordEncoder::WordEncoder
    contextEncoder::ContextEncoder
    output::Linear
    loss::CrossEntropyLoss
end

Classifier(o::Dict,v::Vocabulary) = Classifier(
    WordEncoder(o,v),
    ContextEncoder(o,v),
    Linear(input=o[:hiddenSizes][3]+o[:hiddenSizes][1], output=length(v.comptags)),
    CrossEntropyLoss(dims=1)
)

function loss(M::Classifier, d::SentenceBatch; v::Vocabulary)
    hiddens2 = M.wordEncoder(d.seqInputs, d.seqSizes, d.unsortingIndices; training=true)
    hiddens1 = M.contextEncoder(hiddens2[1]; training=true)
    scores = M.output(vcat(hiddens1[1],hiddens2[1]))
    M.loss(scores,d.compositeOutputs)
end

function predict(M::Classifier, d::SentenceBatch; v::Vocabulary)
    hiddens2 = M.wordEncoder(d.seqInputs, d.seqSizes, d.unsortingIndices; training=false)
    hiddens1 = M.contextEncoder(hiddens2[1]; training=true)
    scores   = M.output(vcat(hiddens1[1],hiddens2[1]))
    preds    = vec(getindex.(argmax(convert(Array,scores),dims=1),1))
    loss     = M.loss(scores,d.compositeOutputs)
    return preds,loss
end

#####
##### Disambiguator Models
#####

abstract type Disambiguator <: Sequential; end;

struct S2SContexDis <: Disambiguator
    wordEncoder::WordEncoder
    contextEncoder::ContextEncoder
    outputEmbed::Embed
    decoder::Decoder
    output::Linear
    loss::CrossEntropyLoss
end

S2SContexDis(o::Dict,v::Vocabulary) = S2SContexDis(
    WordEncoder(o,v),
    ContextEncoder(o,v),
    Embed(input=length(v.tags), output=o[:embedSizes][2]),
    Decoder(o,v),
    Linear(input=o[:hiddenSizes][3], output=length(v.tags)),
    CrossEntropyLoss(dims=1)
)

function loss(M::S2SContexDis, d::SentenceBatch; v::Vocabulary)
    hiddens2 = M.wordEncoder(d.seqInputs, d.seqSizes, d.unsortingIndices; training=true)
    hiddens1 = M.contextEncoder(hiddens2[1]; training=true)
    recurrentLoss(M, d, hiddens1, hiddens2; startIndex=v.specialIndices.bow) / sum(d.masks)
end

function predict(M::S2SContexDis, d::SentenceBatch; v::Vocabulary)
    hiddens2 = M.wordEncoder(d.seqInputs, d.seqSizes, d.unsortingIndices; training=false)
    hiddens1 = M.contextEncoder(hiddens2[1]; training=true)
    recurrentPredict(M, d, hiddens1, hiddens2; v=v)
end

"""
    recurrentPredict(M::Disambiguator, d::SentenceBatch, hiddens1, hiddens2; startIndex=4)

    Make prediciton with recurrent decoder for disambiguation models except MorseDis
"""
function recurrentPredict(M::Disambiguator, d::SentenceBatch, hiddens1, hiddens2; v::Vocabulary)

    preds = Vector{Int}[]
    total = 0.0

    for (w,encodedIO) in enumerate(d.encodedIOs)

        min_loss = typemax(Float64)
        min_loss_seq = Int[];

        for (i,analysis) in enumerate(encodedIO.analyses)

            h1 = (hiddens1[1][:,w:w],hiddens1[2][:,w:w])
            h2 = (hiddens2[1][:,w:w],hiddens2[2][:,w:w])

            seq = [analysis.lemma; analysis.tags; v.specialIndices.eow]

            wtotal = 0.0
            input = v.specialIndices.bow

            for o in seq
                h1, h2 = M.decoder(M.outputEmbed([input]), h1, h2; training=false)
                hh2 = h2[1]
                scores = M.output(reshape(hh2,size(hh2,1),size(hh2,2)))
                wtotal += M.loss(scores,[o])
                input   = o
            end

            loss = wtotal
            if loss < min_loss
                min_loss = loss; min_loss_seq=seq;
            end
        end
        push!(preds,min_loss_seq)
        total+=min_loss
    end

    return PadSequenceArray(preds; pad=v.specialIndices.mask),total
    #TO-DO: total causes incorrect loos in evaluate.jl
end

"""
    MorseDis <: Disambiguator
    MorseDis(config::Dict,v::Vocabulary)
    loss(M::MorseDis, d::SentenceBatch; v::Vocabulary)
    predict(M::MorseDis, d::SentenceBatch; v::Vocabulary)
"""
struct MorseDis <: Disambiguator
    wordEncoder::WordEncoder
    contextEncoder::ContextEncoder
    outputEncoder::OutputEncoder
    outputEmbed::Embed
    decoder::Decoder
    output::Linear
    loss::CrossEntropyLoss
end

MorseDis(o::Dict,v::Vocabulary) = MorseDis(
    WordEncoder(o,v), ContextEncoder(o,v), OutputEncoder(o,v),
    Embed(input=length(v.tags), output=o[:embedSizes][2]),
    Decoder(o,v),
    Linear(input=o[:hiddenSizes][3], output=length(v.tags)),
    CrossEntropyLoss(dims=1)
)

function loss(M::MorseDis, d::SentenceBatch; v::Vocabulary)
    # Word Encoder
    hiddens2 = M.wordEncoder(d.seqInputs, d.seqSizes, d.unsortingIndices; training=true)
    # Context Encoder
    hiddens1 = M.contextEncoder(hiddens2[1]; training=true)
    # output encoder
    esize = size(M.outputEmbed.weight,1);
    tagEmbeddings = [reshape(M.outputEmbed(d.seqOutputs[i,range]),esize,1,length(range)) for (i,range)=enumerate(d.tagRange)]
    hiddens3 = M.outputEncoder(tagEmbeddings)
    # Decoder
    hiddens2 = hiddens2 .+ hiddens3
    recurrentLoss(M, d, hiddens1, hiddens2; startIndex=v.specialIndices.bow)/sum(d.masks)
end

function  predict(M::MorseDis, d::SentenceBatch; v::Vocabulary)
    # Word Encoder
    hiddens2 = M.wordEncoder(d.seqInputs, d.seqSizes, d.unsortingIndices; training=false)
    # Context Encoder
    hiddens1 = M.contextEncoder(hiddens2[1]; training=false)
    # Output Encoder
    hiddens3  = (zero(M.outputEncoder.hidden), zero(M.outputEncoder.hidden))
    # Decoder

    outEmbeddings = [];
    preds = Vector{Int}[]
    total = 0.0

    for (w,encodedIO) in enumerate(d.encodedIOs)

        min_loss = typemax(Float64)
        min_loss_seq::Union{Nothing,Vector{Int}} = nothing;
        min_outEmbed::Union{Nothing,Vector} = nothing;

        for (i,analysis) in enumerate(encodedIO.analyses)

            h2 = (hiddens2[1][:,w:w] .+ hiddens3[1], hiddens2[2][:,w:w] .+ hiddens3[2])
            h1 = (hiddens1[1][:,w:w],hiddens1[2][:,w:w])

            wtotal = 0.0
            input = v.specialIndices.bow
            outEmbedding = [];
            seq = [analysis.lemma; analysis.tags; v.specialIndices.eow]

            for o in seq
                rnninput = M.outputEmbed([input])
                 if input ∉ v.specialIndices && length(v.tags[input])>1
                    push!(outEmbedding,rnninput)
                end
                h1, h2 = M.decoder(rnninput, h1, h2; training=false)
                hh2 = h2[1]
                scores = M.output(reshape(hh2,size(hh2,1),size(hh2,2)))
                wtotal += M.loss(scores,[o])
                input   = o
            end

            loss = wtotal
            if loss < min_loss
                min_loss = loss; min_loss_seq=seq; min_outEmbed=outEmbedding;
            end
        end

        push!(preds,min_loss_seq)
        total += min_loss

        if length(min_outEmbed) != 0

            embedding = hcat(min_outEmbed...)
            push!(outEmbeddings, reshape(embedding, size(embedding,1), 1, size(embedding,2)))

            if length(outEmbeddings) > M.outputEncoder.previous
                popfirst!(outEmbeddings);
            end

            if length(outEmbeddings) > 0
                hiddens3  = (zero(M.outputEncoder.hidden), zero(M.outputEncoder.hidden))
                for embs in outEmbeddings
                    hiddens3 = M.outputEncoder(embs, hiddens3[1], hiddens3[2])
                end
            end
        end
    end
    return PadSequenceArray(preds; pad=v.specialIndices.mask),total
end

#####
##### Utils
#####
function fgbias!(m::LSTM)
    for inputSource in (:i,:h), layer in (:10,:11)
        bias = get(m.gatesview, Symbol(:b,inputSource,:f,layer), nothing)
        if bias !== nothing
            bias .= 0.5
        end
    end
    return m
end

function transfer!(target, tv::Vocabulary, source, sv::Vocabulary)
    for f in (:wordEncoder, :decoder, :contextEncoder, :outputEmbed, :output, :outputEncoder)
        if isdefined(target,f) && isdefined(source,f)
            transfer!(getproperty(target,f), tv, getproperty(source,f), sv)
        end
    end
end

function transfer!(target::Embed, tv::Vocabulary, source::Embed, sv::Vocabulary)
    transfer!(target.weight, tv.tags, source.weight, sv.tags; axis=2)
end

function transfer!(target::Linear, tv::Vocabulary, source::Linear, sv::Vocabulary)
    transfer!(target.mult.weight, tv.tags, source.mult.weight, sv.tags; axis=1)
    transfer!(target.bias, tv.tags, source.bias, sv.tags; axis=1)
end

function transfer!(target, tv::IndexedDict, source, sv::IndexedDict; axis=1)
    for (k,i) in sv.toIndex
        if (j=get(tv,k,nothing)) !== nothing
            if ndims(target) == 2
                if axis == 1
                    target[j,:] = source[i,:]
                else
                    target[:,j] = source[:,i]
                end
            else
                 target[j] = source[i]
            end
        end
    end
end

function transfer!(source::Decoder, sv::Vocabulary, target::Decoder, tv::Vocabulary) 
    transfer!(source.L1,target.L1)
    transfer!(source.L2,target.L2)
end

function transfer!(source::ContextEncoder,sv::Vocabulary,target::ContextEncoder,tv::Vocabulary)
    transfer!(source.encoder,target.encoder)
    transfer!(source.reducer,target.reducer)
end

function transfer!(source::OutputEncoder,sv::Vocabulary,target::OutputEncoder,tv::Vocabulary)
    transfer!(source.encoder,target.encoder)
end

function transfer!(target::WordEncoder, tv::Vocabulary, source::WordEncoder, sv::Vocabulary)
    transfer!(target.embed.weight, tv.chars, source.embed.weight, sv.chars; axis=2)
    transfer!(target.encoder, source.encoder)
end

function transfer!(target::Linear, source::Linear) 
    transfer!(target.mult,source.mult)
    transfer!(target.bias,source.bias)
end
transfer!(target::Multiply, source::Multiply) = transfer!(target.weight,source.weight)
transfer!(target::Dense, source::Dense) = transfer!(target.linear,source.linear)
transfer!(target::LSTM, source::LSTM) = transfer!(target.params,source.params)
transfer!(target::Param, source::Param) = copyto!(value(target),value(source))