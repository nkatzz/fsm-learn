sequence(X) :- seq(X,_,_).
step(X) :- seq(_,_,X).

inState(SeqId,1,1) :- sequence(SeqId).
inState(SeqId,S2,T+1) :- inState(SeqId,S1,T), transition(S1,Symbol,S2), seq(SeqId,Symbol,T).
accepted(SeqId) :- inState(SeqId,S,T), seqEnd(SeqId,T), accepting(S).
% accepted(SeqId) :- inState(SeqId,S,T), accepting(S).

% partiallyParsed(SeqId,T) :- inState(SeqId,S,T), accepting(S).
% accepted(SeqId,T) :- #false: partiallyParsed(SeqId,Z), Z < T ; partiallyParsed(SeqId,T).

% :- partiallyParsed(SeqId,T), inState(SeqId,_,T1), T1 > T.
% accepted(SeqId,T) :- partiallyParsed(SeqId,T).

% accepted(SeqId) :- partiallyParsed(SeqId,T), step(T).

#program seq_end.
seq(SeqId,T) :- seq(SeqId,_,T).
seqEnd(SeqId,T) :- seq(SeqId,T), not seq(SeqId,T1), T1 = T+1.

#program seq_end_inc(k).
seqEnd(SeqId,k) :- sequence(SeqId).





