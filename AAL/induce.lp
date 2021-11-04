% #include "symbols.lp".

#include "symbols_1.lp".

#include "interpreter.lp".
#include "classes.lp".
% #include "countingChains.lp".

%----------------------------------------
#program states(n).
statesEnum(1..n).

%----------------------------------------
#program constraints_exact.
:- positive(SeqId), not accepted(SeqId).
:- negative(SeqId), accepted(SeqId).

%----------------------------------------
#program constraints_heuristic.
%:~ falsePositive(SeqId,T). [T@2,SeqId]
%:~ truePositive(SeqId,T). [-T@2,SeqId]

% #maximize{1@4, Seq: positive(Seq), accepted(Seq)}.
#minimize{1@3,Seq: positive(Seq), not accepted(Seq)}.
#minimize{1@2,Seq: negative(Seq), accepted(Seq)}.

%----------------------------------------
#program base.
% state(1). state(2). state(3).  state(4). state(5).

:- transition(S,_,S2), accepting(S), S2 != S.
:- transition(S1,X,S2), transition(S1,X,S3), S2 != S3.

{ state(X) } :- statesEnum(X).
{ accepting(X) }  :- state(X).

{ at_most(S1,X,S2) } :- state(S1), state(S2), symbol(X).%, S1 <= S2.
{ at_least(S1,X,S2) } :- state(S1), state(S2), symbol(X).% S1 <= S2.
%{ transition(S1,X,S2) } :- state(S1), state(S2), symbol(X).

% implicitly generate self-loops where appropriate.
% transition(S,X,S) :- not transition(S,X,S1), state(S1), state(S), symbol(X), S != S1.

#minimize{1@1,S1,S2,X: at_most(S1,X,S2)}.
#minimize{1@1,S1,S2,X: at_least(S1,X,S2)}.
#minimize{1@1,S1,S2,X: transition(S1,X,S2)}.

% :- transition(S,X,S), accepting(S). % I was using this while trying stuff for partially parsing.

% #minimize{T@0,SeqId,T: accepted(SeqId,T), positive(SeqId)}.

% #minimize{1@1,S1,S2,X: transition(S1,X,S2)}.
% #minimize{-Z@1,S1,S2,X: at_most(S1,X,S2), cost(X,Z)}. % larger value is more general for at_most, i.e. at_most(i) implies at_most(a)
% #minimize{Z@1,S1,S2,X: at_least(S1,X,S2), cost(X,Z)}. % smaller value is more general for at_least, i.e at_most(a) implies at_most(k)

% #show transition_1/3.
% #show move/3.
%#show transition/3.
#show at_most/3.
#show at_least/3.
% #show countingTransition/3.
% #show neg/3.
#show accepting/1.





