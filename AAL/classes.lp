#program class(k).
targetClass(k).

#program classes_defs.
positive(SeqId) :- class(SeqId,X), targetClass(X).
negative(SeqId) :- class(SeqId,X), not targetClass(X).

% falseNegative(SeqId) :- positive(SeqId), not accepted(SeqId).
% falsePositive(SeqId) :- negative(SeqId), accepted(SeqId).
% truePositive(SeqId) :- positive(SeqId), accepted(SeqId).

falseNegative(SeqId) :- positive(SeqId), not accepted(SeqId).

%falsePositive(SeqId,T) :- negative(SeqId), accepted(SeqId,T).
%truePositive(SeqId,T) :- positive(SeqId), accepted(SeqId,T).

falsePositive(SeqId,49) :- negative(SeqId), accepted(SeqId).
truePositive(SeqId,49) :- positive(SeqId), accepted(SeqId).