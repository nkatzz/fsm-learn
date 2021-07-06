#script (python)
import string
import clingo

def get_symbol(symbol_index):
    x = int(str(symbol_index))
    alphabet = list(string.ascii_lowercase)
    return clingo.Function(alphabet[x - 1], [])

#end.

alphabetNum(1..20).
_symbol(@get_symbol(Y)) :- alphabetNum(Y).

symbol(feature1(X)) :- _symbol(X).

lesseq(a,b). lesseq(b,c). lesseq(c,d). lesseq(d,e). lesseq(e,f). lesseq(f,g). lesseq(g,h). % this is for alphabet size = 8
lesseq(h,i). lesseq(i,j). lesseq(j,k). lesseq(k,l). lesseq(l,m). lesseq(m,n). lesseq(n,o).
lesseq(o,p). lesseq(p,q). lesseq(q,r). lesseq(r,s). lesseq(s,t).

lesseq(X,X) :- _symbol(X).
lesseq(X,Y) :- lesseq(X,Z), lesseq(Z,Y).

lesseq1(feature1(X),feature1(Y)) :- lesseq(X,Y).

transition(S1,X,S2) :- at_most(S1,Y,S2), lesseq1(X,Y).
transition(S1,X,S2) :- at_least(S1,Y,S2), lesseq1(Y,X).