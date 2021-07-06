#script (python)
import string
import clingo

def get_symbol(symbol_index):
    x = int(str(symbol_index))
    alphabet = list(string.ascii_lowercase)
    return clingo.Function(alphabet[x - 1], [])

#end.

alphabetNum(1..8).
_symbol(@get_symbol(Y)) :- alphabetNum(Y).

symbol(alive(X)) :- _symbol(X).
symbol(necrotic(X)) :- _symbol(X).
symbol(apoptotic(X)) :- _symbol(X).

lesseq(a,b). lesseq(b,c). lesseq(c,d). lesseq(d,e). lesseq(e,f). lesseq(f,g). lesseq(g,h).

lesseq(X,X) :- _symbol(X).
lesseq(X,Y) :- lesseq(X,Z), lesseq(Z,Y).

lesseq1(alive(X),alive(Y)) :- lesseq(X,Y).
lesseq1(necrotic(X),necrotic(Y)) :- lesseq(X,Y).
lesseq1(apoptotic(X),apoptotic(Y)) :- lesseq(X,Y).

transition(S1,X,S2) :- at_most(S1,Y,S2), lesseq1(X,Y).
transition(S1,X,S2) :- at_least(S1,Y,S2), lesseq1(Y,X).