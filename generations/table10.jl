using Pkg; Pkg.activate("../")
using Morse
import Morse: CharDict, StrDict

function CounterVocabulary(sets::Vector)
    char2ix, tag2ix, word2ix, ctag2ix = CharDict(), StrDict(), StrDict(), StrDict()

    for (i,T) in enumerate(specialTokens)
        get!(tag2ix,T,1); get!(word2ix,T,1); get!(word2ix,T,1)
        get!(char2ix,T[1],1);
    end
    specialIndicies = (unk=1, mask=1, eow=1, bow=1)

    for data in sets, sentence in data, par::ParsedIO in sentence
        for c::Char in par.chars
            char2ix[c] = get(char2ix, c, 0) + 1
        end
        for a::Analysis in par.analyses
            for c::Char in a.lemma
                tag2ix[string(c)] = get(tag2ix, string(c), 0) + 1
            end
            ctag2ix[join(a.tags,'+')] = get(ctag2ix,join(a.tags,'+'),0) + 1
        end
        word2ix[join(par.chars)] =  get(word2ix,join(par.chars),0) + 1
    end

    for data in sets, sentence in data, par::ParsedIO in sentence
        for a::Analysis in par.analyses
            for t::String in a.tags
                tag2ix[t] = get(tag2ix,t,0) + 1
            end
        end
    end
    return (chars=char2ix, tags=tag2ix, ctags=ctag2ix, words=word2ix)
end

config = Morse.intro(ARGS)


p_cls = string("morse.Classifier_lemma_",config[:lemma],"_lang_UD-",config[:langcode],"_size_full.gen")
p_seq = string("morse.S2SContext_lemma_",config[:lemma],"_lang_UD-",config[:langcode],"_size_full.gen")

DataSet  = eval(Meta.parse(config[:dataSet]))
ModelType = eval(Meta.parse(config[:modelType]))

code = config[:langcode]
lang = CODE_TO_LANG[code]
files = Morse.dir("data","ud-treebanks-v2.1/UD_$(lang)") * "/$(code)-ud-" .* ["train","dev","test"] .* ".conllu"

parser     = Parser{DataSet}(Val(config[:version]))
trn        = parseFile(first(files); p=parser, withLemma=config[:lemma], parseAll=false)
tst        = parseFile(last(files); p=parser, withLemma=config[:lemma], parseAll=false) 

trncounter = CounterVocabulary([trn])

tst_pred_seq = parseFile(p_seq; p=parser, withLemma=config[:lemma], parseAll=false)
tst_pred_cls = parseFile(p_cls; p=parser, withLemma=config[:lemma], parseAll=false)

@assert length(tst_pred_seq) == length(tst) "seq gens broken"
@assert length(tst_pred_cls) == length(tst) "classification gens broken $(length(tst_pred_cls)) != $(length(tst))"
counter = trncounter.ctags

function stats(tst,seq,cls,counter)
    totaltokens = [0,0,0]
    seqtrues = [0,0,0]
    clstrues = [0,0,0]
    
    for (s,sent) in enumerate(tst)
        for (i,wordio) in enumerate(sent)
            if wordio.analyses[1].isValid
                correct_analysis = join(wordio.analyses[1].tags,'+')
                cnt = get(counter,correct_analysis,0)
                seq_cnt   = join(seq[s][i].analyses[1].tags,'+')  == correct_analysis
                cls_cnt   = join(cls[s][i].analyses[1].tags,'+')  == correct_analysis
                
                if cnt == 0
                    totaltokens[1] += 1
                    seqtrues[1]    += seq_cnt
                    clstrues[1]    += cls_cnt 
                elseif cnt < 5
                    totaltokens[2] += 1
                    seqtrues[2]    += seq_cnt
                    clstrues[2]    += cls_cnt 
                else cnt >= 5
                    totaltokens[3] += 1
                    seqtrues[3]    += seq_cnt
                    clstrues[3]    += cls_cnt 
                end
            end
        end
    end
    return totaltokens, seqtrues, clstrues
end

total, seqt, clst =stats(tst,tst_pred_seq,tst_pred_cls,counter)


println(uppercase(code)," & ",total[1],"/",total[2],"/",total[3]," &\t ",100clst[1]/total[1]," & ",100seqt[1]/total[1],
        " &\t ",100clst[2]/total[2]," & ",100seqt[2]/total[2],
        " &\t ",100clst[3]/total[3]," & ",100seqt[3]/total[3])

# function pred_upos_trim!(data)
#     for sent in data
#         for wordio in sent
#             analysis = wordio.analyses[1]
#             if startswith(analysis.tags[1],"Upos=Upos=")
#                 analysis.tags[1] = String(SubString(analysis.tags[1],6:length(analysis.tags[1])))
#             end
#         end
#     end
# end


#pred_upos_trim!(tst_pred_seq)
#pred_upos_trim!(tst_pred_cls)

# @show length(tst)
# @show length(tst_pred_seq)
# @show length(tst_pred_cls)
# @show last(tst_pred_seq)[1]
# @show last(tst)[1]
