learning_program = """
symbol(X) :- seq(_,X,_).
sequence(X) :- seq(X,_,_).

inState(SeqId,1,1) :- sequence(SeqId).
inState(SeqId,S2,T2) :- inState(SeqId,S1,T1), T2 = T1+1, transition(S1,Symbol,S2), seq(SeqId,Symbol,T1).
parsed(SeqId) :- inState(SeqId,S,T), seqEnd(SeqId,T), final(S).

positive(SeqId) :- class(SeqId,1).
negative(SeqId) :- class(SeqId,0).

seq(SeqId,T) :- seq(SeqId,_,T).
seqEnd(SeqId,T) :- seq(SeqId,T), not seq(SeqId,T1), T1 = T+1.

statesEnum(1..2).
state(1).

{ state(X) } :- statesEnum(X).
{ final(X) } :- state(X).
{ transition(S1,X,S2) } :- state(S1), state(S2), symbol(X).

% Complete automata:
%:- not transition(S,X,_), symbol(X), state(S). 

% Deterministic automata:
:- transition(S1,X,S2), transition(S1,X,S3), S2 != S3.

falseNegative(SeqId) :- positive(SeqId), not parsed(SeqId).
falsePositive(SeqId) :- negative(SeqId), parsed(SeqId).

:- falseNegative(SeqId).
:- falsePositive(SeqId).

%:~ falseNegative(SeqId). [1@2,SeqId]
%:~ falsePositive(SeqId). [1@2,SeqId]

%#minimize{1@1,S1,X,S2: transition(S1,X,S2)}.

#show transition/3.
#show final/1.
"""

inference_program = """
symbol(X) :- seq(_,X,_).
sequence(X) :- seq(X,_,_).

inState(SeqId,s2,1) :- sequence(SeqId).
inState(SeqId,S2,T2) :- inState(SeqId,S1,T1), T2 = T1+1, transition(S1,Symbol,S2), seq(SeqId,Symbol,T1).
parsed(SeqId) :- inState(SeqId,S,T), seqEnd(SeqId,T), final(S).

seq(SeqId,T) :- seq(SeqId,_,T).
seqEnd(SeqId,T) :- seq(SeqId,T), not seq(SeqId,T1), T1 = T+1.

positive(SeqId) :- class(SeqId,1).
negative(SeqId) :- class(SeqId,0).

falseNegative(SeqId) :- positive(SeqId), not parsed(SeqId).
falsePositive(SeqId) :- negative(SeqId), parsed(SeqId).

fns(X) :- X = #count{SeqId: falseNegative(SeqId)}.
fps(X) :- X = #count{SeqId: falsePositive(SeqId)}.
tps(X) :- X = #count{SeqId: positive(SeqId), parsed(SeqId)}.

#show fns/1.
#show fps/1.
#show tps/1.
"""

learning_program_ec = """

symbol(X) :- seq(_,X,_).
sequence(X) :- seq(X,_,_).
step(X) :- seq(_,_,X).

holdsAt(F,T2) :- initiatedAt(F,T1), T2 = T1+1, fluent(F), step(T1), step(T2).
holdsAt(F,T2) :- holdsAt(F,T1), not terminatedAt(F,T1), T2 = T1+1, fluent(F), step(T1), step(T2).
holdsAt(F,1) :- initially(F).

fluent(inState(SeqId,State)) :- sequence(SeqId), state(State). 
initially(inState(SeqId,1)) :- sequence(SeqId).

initiatedAt(inState(SeqId,S2),T) :- holdsAt(inState(SeqId,S1),T), transition(S1,Symbol,S2), seq(SeqId,Symbol,T).
terminatedAt(inState(SeqId,S1),T) :- initiatedAt(inState(SeqId,S2),T), S1 != S2, state(S1), state(S2).
parsed(SeqId) :- holdsAt(inState(SeqId,State),T), seqEnd(SeqId,T), final(State).

seq(SeqId,T) :- seq(SeqId,_,T).
seqEnd(SeqId,T) :- seq(SeqId,T), not seq(SeqId,T1), T1 = T+1.

%:- holdsAt(inState(SeqId,S1),T),  holdsAt(inState(SeqId,S2),T), S1 != S2.
%:- transition(S1,X,S2), transition(S1,X,S3), S2 != S3. % Not useful.

statesEnum(1..2).
state(1).

{ state(X) } :- statesEnum(X).
{ final(X) } :- state(X).
{ transition(S1,X,S2) } :- state(S1), state(S2), symbol(X).

%:- not transition(S,X,_), symbol(X), state(S).

positive(SeqId) :- class(SeqId,1).
negative(SeqId) :- class(SeqId,0).
%negative(SeqId) :- class(SeqId,X), X != 1.

falseNegative(SeqId) :- positive(SeqId), not parsed(SeqId).
falsePositive(SeqId) :- negative(SeqId), parsed(SeqId).

:- falseNegative(SeqId).
:- falsePositive(SeqId).

%:~ falseNegative(SeqId). [1@2,SeqId]
%:~ falsePositive(SeqId). [1@2,SeqId]

#minimize{1@1,S1,X,S2: transition(S1,X,S2)}.

#show transition/3.
#show final/1.
"""

inference_program_ec = """
symbol(X) :- seq(_,X,_).
sequence(X) :- seq(X,_,_).
step(X) :- seq(_,_,X).

holdsAt(F,T2) :- initiatedAt(F,T1), T2 = T1+1, fluent(F), step(T1), step(T2).
holdsAt(F,T2) :- holdsAt(F,T1), not terminatedAt(F,T1), T2 = T1+1, fluent(F), step(T1), step(T2).
holdsAt(F,1) :- initially(F).

initiatedAt(inState(SeqId,S2),T2) :- holdsAt(inState(SeqId,S1),T1), S1 != S2, T2 = T1+1, transition(S1,Symbol,S2), seq(SeqId,Symbol,T1).  
terminatedAt(inState(SeqId,S1),T) :- initiatedAt(inState(SeqId,S2),T), S1 != S2, state(S1), state(S2).
parsed(SeqId) :- holdsAt(inState(SeqId,State),T), seqEnd(SeqId,T),final(State).

fluent(inState(SeqId,State)) :- sequence(SeqId), state(State). 
initially(inState(SeqId,1)) :- sequence(SeqId).

state(X) :- transition(X,_,_).
state(X) :- transition(_,_,X).

seq(SeqId,T) :- seq(SeqId,_,T).
seqEnd(SeqId,T) :- seq(SeqId,T), not seq(SeqId,T1), T1 = T+1.

positive(SeqId) :- class(SeqId,1).
negative(SeqId) :- class(SeqId,0).

falseNegative(SeqId) :- positive(SeqId), not parsed(SeqId).
falsePositive(SeqId) :- negative(SeqId), parsed(SeqId).

fns(X) :- X = #count{SeqId: falseNegative(SeqId)}.
fps(X) :- X = #count{SeqId: falsePositive(SeqId)}.
tps(X) :- X = #count{SeqId: positive(SeqId), parsed(SeqId)}.

#show fns/1.
#show fps/1.
#show tps/1.
"""