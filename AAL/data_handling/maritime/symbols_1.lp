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

symbol(longitude(X)) :- _symbol(X).
symbol(latitude(X)) :- _symbol(X).
symbol(speed(X)) :- _symbol(X).
symbol(heading(X)) :- _symbol(X).
symbol(course_over_ground(X)) :- _symbol(X).

lesseq(a,b). lesseq(b,c). lesseq(c,d). lesseq(d,e). lesseq(e,f). lesseq(f,g). lesseq(g,h).

lesseq(X,X) :- _symbol(X).
lesseq(X,Y) :- lesseq(X,Z), lesseq(Z,Y).

lesseq1(longitude(X),longitude(Y)) :- lesseq(X,Y).
lesseq1(latitude(X),latitude(Y)) :- lesseq(X,Y).
lesseq1(speed(X),speed(Y)) :- lesseq(X,Y).
lesseq1(heading(X),heading(Y)) :- lesseq(X,Y).
lesseq1(course_over_ground(X),course_over_ground(Y)) :- lesseq(X,Y).

transition(S1,X,S2) :- at_most(S1,Y,S2), lesseq1(X,Y).
transition(S1,X,S2) :- at_least(S1,Y,S2), lesseq1(Y,X).