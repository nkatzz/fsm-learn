countingState(5..20).
state(5).
state(20).

{ transition_1(S1,X,S2) } :- countingState(S1), countingState(S2), S2 = S1+1, symbol(X).
{ transition_1(S1,X,S2) } :- countingState(S1), accepting(S2), symbol(X). % allow jumps to final state.
{ transition_1(S,X,S) } :- countingState(S), symbol(X).                   % allow self-loops in counting states.

%:- transition_1(S1,X,S2), not transition(_,Y,S1), symbol(Y). % forbid counting transitions that are not linked to the rest of the FSM.

transition(S1,X,S2) :- transition_1(S1,X,S2).
#minimize{1@1,S1,S2,X: transition_1(S1,X,S2)}.


