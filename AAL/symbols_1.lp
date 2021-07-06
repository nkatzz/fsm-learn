#script (python)
import string
import clingo

def get_symbol(symbol_index):
    x = int(str(symbol_index))
    alphabet = list(string.ascii_lowercase)
    return clingo.Function(alphabet[x - 1], [])

#end.

alphabetNum(1..20).
symbol(@get_symbol(Y)) :- alphabetNum(Y).

lesseq(a,b). lesseq(b,c). lesseq(c,d). lesseq(d,e). lesseq(e,f). lesseq(f,g). lesseq(g,h).

lesseq(X,X) :- symbol(X).
lesseq(X,Y) :- lesseq(X,Z), lesseq(Z,Y).

% symbol(any).
% transition(S1,X,S2) :- move(S1,X,S2).
% transition(S1,X,S2) :- move(S1,any,S2), symbol(X).

transition(S1,X,S2) :- at_most(S1,Y,S2), lesseq(X,Y).
transition(S1,X,S2) :- at_least(S1,Y,S2), lesseq(Y,X).

:- at_most(S1,X,S2), at_most(S1,Y,S2), lesseq(Y,X), Y != X.
:- at_least(S1,X,S2), at_least(S1,Y,S2), lesseq(X,Y), Y != X.

% transition(S1,X,S2) :- neg(S1,Y,S2), symbol(Y), symbol(X), X != Y.

%symbol(bigIncrease).
%symbol(smallIncrease).
%symbol(noChange).

%transition(S1,bigIncrease,S2) :- inState(SeqId,S1,T), seq(SeqId,X,T), inState(SeqId,S2,T+1), seq(SeqId,X,T),