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

symbol(feature1(X)) :- _symbol(X).
symbol(feature2(X)) :- _symbol(X).
symbol(feature3(X)) :- _symbol(X).
symbol(feature4(X)) :- _symbol(X).
symbol(feature5(X)) :- _symbol(X).
symbol(feature6(X)) :- _symbol(X).


lesseq(a,b). lesseq(b,c). lesseq(c,d). lesseq(d,e). lesseq(e,f). lesseq(f,g). lesseq(g,h).

lesseq(X,X) :- _symbol(X).
lesseq(X,Y) :- lesseq(X,Z), lesseq(Z,Y).

lesseq1(feature1(X),feature1(Y)) :- lesseq(X,Y).
lesseq1(feature2(X),feature2(Y)) :- lesseq(X,Y).
lesseq1(feature3(X),feature3(Y)) :- lesseq(X,Y).
lesseq1(feature4(X),feature4(Y)) :- lesseq(X,Y).
lesseq1(feature5(X),feature5(Y)) :- lesseq(X,Y).
lesseq1(feature6(X),feature6(Y)) :- lesseq(X,Y).

transition(S1,X,S2) :- at_most(S1,Y,S2), lesseq1(X,Y).
transition(S1,X,S2) :- at_least(S1,Y,S2), lesseq1(Y,X).