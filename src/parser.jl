import KnetLayers: IndexedDict

#####
##### Helper Functions for Base
#####

### TO-DO: Localization for all languages
"""
    TRtoLower(x::Char)
    TRtoLower(s::AbstractString)

    lowercase function for Turkish locale
"""
TRtoLower(x::Char) = x=='I' ? 'ı' : lowercase(x);
TRtoLower(s::AbstractString) = map(TRtoLower,s)


#####
##### IO and Vocabulary
#####

"""
    const specialTokens = (unk="❓", mask="⭕️", eow="🏁", bow="🎬")
    unk: unknown inputs
    mask: mask token
    eow: end of word/sentence
    bow: beginning of word/sentence
"""
const specialTokens = (unk="❓", mask="⭕️", eow="🏁", bow="🎬")

"""
    Vocabulary
    Vocabulary(sets::Array)

    General Vocabulary structure for models presented in Morse.

    Vocabulary can be created from sets of ParsedIO arrays.
"""
struct Vocabulary
    chars::IndexedDict{Char}
    tags::IndexedDict{String}
    comptags::IndexedDict{String}
    words::IndexedDict{String}
    specialTokens::NamedTuple
    specialIndices::NamedTuple
end

"""
    Analysis

    Analysis is structure that keeps morphological anlysis of a word
"""
struct Analysis
    lemma::Vector{Char}
    tags::Vector{String}
    isValid::Bool
end

"""
    EncodedAnalysis

    Encoded Analysis is structure that keeps morphological anlysis
    of a word with encoded fields.
"""
struct EncodedAnalysis
    lemma::Vector{Int}
    tags::Vector{Int}
    compositeTag::Int
    isValid::Bool
end

EncodedAnalysis(a::Analysis; v::Vocabulary) =
EncodedAnalysis(map(x->get(v.tags, string(x), v.specialIndices.unk)::Int, a.lemma),
                map(x->get(v.tags, x, v.specialIndices.unk)::Int, a.tags),
                get(v.comptags, join(a.tags,'|'), v.specialIndices.unk),
                a.isValid)

"""
    ParsedIO
    parseDataLine(line::AbstractString, p::Parser{MyDataSet}; wLemma=true, parseAll=false)

    ParsedIO keeps input and output of a single dataset line
    Input are `chars` of the word.
    Outputs are all possible `Analysis`s of tht word

    ParsedIO can be created from a dataset line, with `parseDataLine`
"""
struct ParsedIO
    chars::Vector{Char}
    analyses::Vector{Analysis}
    isAmbigous::Bool
end

"""
    EncodedIO
    EncodedIO(a::ParsedIO; v::Vocabulary)

    EncodedIO keeps input and output of a single dataset line
    Input are one hot encodings of the `chars` of the word.
    Outputs are all possible `EncodedAnalysis`s of that word

    `EncodedIO` can be created from a `ParsedIO` via a `Vocabulary`
"""
struct EncodedIO
    chars::Vector{Int}
    analyses::Vector{EncodedAnalysis}
    completeWord::Int
    isAmbigous::Bool
end

EncodedIO(a::ParsedIO; v::Vocabulary) =
    EncodedIO(map(x->get(v.chars, x, v.specialIndices.unk)::Int, a.chars),
              map(x->EncodedAnalysis(x;v=v), a.analyses),
              get(v.words, join(a.chars), v.specialIndices.unk),
              a.isAmbigous)

encode(data::Vector; v::Vocabulary) = encode.(data,v=v)
encode(s::Vector{ParsedIO}; v::Vocabulary) = EncodedIO.(s,v=v)

const StrDict  = Dict{String,Int}
const CharDict = Dict{Char,Int}

function Vocabulary(sets::Vector)
    char2ix, tag2ix, word2ix, ctag2ix=CharDict(), StrDict(), StrDict(), StrDict()

    for (i,T) in enumerate(specialTokens)
        get!(tag2ix,T,i); get!(word2ix,T,i); get!(word2ix,T,i)
        get!(char2ix,T[1],i);
    end
    specialIndicies = (unk=1, mask=2, eow=3, bow=4)

    for data in sets, sentence in data, par::ParsedIO in sentence
        for c::Char in par.chars
            get!(char2ix, c, length(char2ix)+1)
        end
        for a::Analysis in par.analyses
            for c::Char in a.lemma
                get!(tag2ix, string(c), length(tag2ix)+1)
            end
            get!(ctag2ix,join(a.tags,'+'),length(ctag2ix)+1)
        end
        get!(word2ix,join(par.chars),length(word2ix)+1)
    end

    for data in sets, sentence in data, par::ParsedIO in sentence
        for a::Analysis in par.analyses
            for t::String in a.tags
                get!(tag2ix,t,length(tag2ix)+1)
            end
        end
    end

    Vocabulary(IndexedDict(char2ix), IndexedDict(tag2ix),
               IndexedDict(ctag2ix), IndexedDict(word2ix),
               specialTokens, specialIndicies)
end

#####
##### DataSet and Parser
#####

"""
    DataSet

    Abstract type for datasets used in Morse
"""
abstract type DataSet; end
"""
    UDDataSet

    Universal Dependencies Dataset
"""
abstract type UDDataSet <: DataSet; end

"""
    TRDataSet

    TrMor dataset
"""
abstract type TRDataSet <: DataSet; end


const NString  = Union{Nothing,String}

"""
    Parser{T<:DataSet}

    Parser structure for datasets.
"""
struct Parser{MyDataSet}
    sentenceStart::String
    sentenceEnd::String
    partsSeperator::Char
    tagsSeperator::Char
    skipLines::Vector{String}
    unkToken::String
    dbToken::NString # derivational boundry
    tagValueSeperator::NString
end

Parser{UDDataSet}(v=21) =
    Parser{UDDataSet}("# text", " ", '\t', '|', ["#"],"_", nothing, "=")

Parser{TRDataSet}(v=2018) =
    Parser{TRDataSet}("<S", "</S", (v==2018 ? '\t' : ' '), '+', ["<"], "*UNKNOWN*", "^DB", nothing)

"""
     parseDataLine(line::AbstractString, p::Parser{<:MyDataSet}; wLemma=true, parseAll)

     It parses a line from given dataset. Returns a ParsedIO object.

Keywords
========
     * wLemma = decision for adding lemma to the outputs.
     * parseAll = decision for adding all posible anlyses or the correct one
"""
function parseDataLine(line::AbstractString, p::Parser{TRDataSet}; wLemma=true, parseAll=false)
    tokens = split(line, p.partsSeperator)
    ParsedIO(collect(TRtoLower(tokens[1])),
             map(t->Analysis(t, p; wLemma=wLemma), tokens[2:(parseAll ? end : 2)]),
             length(tokens)>2);
end

function Analysis(analysis::AbstractString, p::Parser{TRDataSet}; wLemma=true)
    if occursin(p.unkToken,analysis) #unknown case
        lemma   = ""
        isValid = false
        tags    = String[specialTokens.unk]
    elseif occursin(p.tagsSeperator^2,analysis) # ++ case
        lemma   = string(p.tagsSeperator)
        isValid = true
        tags = String[split(analysis,p.tagsSeperator^2)[2]]
    else
        tokens  = split(analysis,p.tagsSeperator)
        lemma   = popfirst!(tokens)
        isValid	= true
        tags    = String[]
        for tag in tokens
            if endswith(tag, p.dbToken)
                push!(tags, tag[1:end-length(p.dbToken)], p.dbToken)
            else
                push!(tags, tag)
            end
        end
    end    
    lemmaOut = wLemma ? collect(TRtoLower(lemma)) : Char[]
    return Analysis(lemmaOut, tags, isValid)
end

function parseDataLine(line::AbstractString, p::Parser{UDDataSet}; wLemma=true, parseAll=false)
    tokens = split(line,p.partsSeperator)
    feats = tokens[6]
    occursin("-",first(tokens)) && return nothing
    if feats != p.unkToken
        tags = pushfirst!(split(feats,p.tagsSeperator),"Upos="*tokens[4])
    else
        tags = String["Upos="*tokens[4]]
    end
    lemmaOut = wLemma ? collect(lowercase(tokens[3])) : Char[]
    return ParsedIO(collect(lowercase(tokens[2])), [Analysis(lemmaOut , tags, true)], false)
end

"""
     parseFile(file::AbstractString; p::Parser, withLemma=true, parseAll)

It parses a file from given dataset. Returns a ParsedIO array.

Keywords
========
     * withLemma = decision for adding lemma to the outputs.
     * parseAll = decision for adding all posible anlyses or the correct one
"""
function parseFile(file::AbstractString; p::Parser, withLemma=true, parseAll=false)
    data = []
    sentence = ParsedIO[]
    for line in eachline(file)
        if startswith(line,p.sentenceStart)
            sentence = ParsedIO[]
        elseif isempty(line) || startswith(line,p.sentenceEnd)
           isempty(sentence) && continue # push!(data,sentence) #continue
           if any(a->a.analyses[1].isValid,sentence)
               push!(data,sentence)
           end
        elseif !any(startswith.(line,p.skipLines))
            parsedLine = parseDataLine(line, p; wLemma=withLemma, parseAll=parseAll)
            parsedLine !== nothing && push!(sentence,parsedLine)
        end
    end
    return data
end
