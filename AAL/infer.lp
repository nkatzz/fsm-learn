#include "symbols_1.lp".
#include "interpreter.lp".
#include "classes.lp".

fns(X) :- X = #count{SeqId: falseNegative(SeqId)}.
fp(SeqId,T,SeqLength) :- falsePositive(SeqId,T), seqEnd(SeqId,SeqLength). % seqEnd/2 is defined in interpreter.lp
tp(SeqId,T,SeqLength) :- truePositive(SeqId,T),  seqEnd(SeqId,SeqLength).

#show fns/1.
#show fp/3.
#show tp/3.