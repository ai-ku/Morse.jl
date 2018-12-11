"""
    SentenceBatch

    Keeps input/outputs of a sentence with encoded fields.
"""
struct SentenceBatch
    encodedIOs::Vector{EncodedIO}
    seqInputs::Vector{Int}
    seqSizes::Vector{Int}
    seqOutputs::Matrix{Int}
    masks::Matrix{Int}
    tagRange::Vector{UnitRange{Int}}
    unsortingIndices::Vector{Int}
    wordInputs::Vector{Int} # words
    compositeOutputs::Vector{Int}
end

"""
    miniBatch(data, v::Vocabulary; sorted=false)
    miniBatch(sentence::Vector{EncodedAnalysis}, v::Vocabulary)

    Creates SentenceBatche given encodedIO array for a sentence
"""
miniBatch(data, v::Vocabulary; sorted=false) =
map(d->miniBatch(d,v), sorted ? sort(data; by=length) : data)

function miniBatch(sentence::Vector{EncodedIO}, v::Vocabulary)
    #complete inputs
    wordInputs = map(a->a.completeWord,sentence)
    #sequence inputs
    seqInputs, seqSizes, unsortingIndices = sequenceInputBatch(sentence)
    #get correct analyses
    analyses = map(a->first(a.analyses),sentence)
    #composite output
    compositeOutputs = map(a->a.compositeTag,analyses)
    #sequence outputs
    seqOutputs, masks, tagRange = sequenceOutputBatch(analyses, v)
    SentenceBatch(sentence, seqInputs, seqSizes, seqOutputs, masks, tagRange,
                            unsortingIndices, wordInputs, compositeOutputs)
end

function sequenceInputBatch(sentence::Vector{EncodedIO})
    #sequence input
    words = map(a->a.chars,sentence)
    sortingIndices = sortperm(length.(words); rev=true)
    unsortingIndices = sortperm(sortingIndices)
    seqInputs, seqSizes = _pack_sequence(words[sortingIndices])
    return seqInputs, seqSizes, unsortingIndices
end

function sequenceOutputBatch(analyses::Vector{EncodedAnalysis}, v::Vocabulary)
    #sequence output
    lengths  = map(a->length(a.lemma)+length(a.tags)+1,analyses)
    tagRange = map(a->length(a.lemma)+1:length(a.lemma)+length(a.tags), analyses)
    maxoutlen = maximum(lengths)
    seqOutputs = fill(v.specialIndices.mask, length(analyses), maxoutlen)
    masks = ones(Int, length(analyses), maxoutlen)
    @inbounds for (k,a) in enumerate(analyses)
        seqOutputs[k,1:length(a.lemma)] .= a.lemma
        seqOutputs[k,tagRange[k]] .= a.tags
        seqOutputs[k,lengths[k]] = v.specialIndices.eow
        if a.isValid
            masks[k,lengths[k]+1:maxoutlen] .= 0
        else
            masks[k,:] .= 0
        end
    end
    return seqOutputs, masks, tagRange
end
