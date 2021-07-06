transition(1,b,1). transition(1,f,1). transition(1,g,1). transition(1,i,1). transition(1,j,1). transition(1,k,1). transition(1,m,1). transition(1,n,1). transition(1,o,1). transition(1,p,1). transition(1,q,1). transition(1,r,1). transition(1,t,1). transition(1,a,2). transition(1,c,2). transition(1,d,2). transition(1,e,2). transition(1,f,2). transition(1,g,2). transition(1,h,2). transition(1,i,2). transition(1,k,2). transition(1,l,2). transition(1,m,2). transition(1,o,2). transition(1,s,2). transition(1,t,2). transition(2,a,1). transition(2,b,1). transition(2,c,1). transition(2,d,1). transition(2,f,1). transition(2,h,1). transition(2,i,1). transition(2,j,1). transition(2,k,1). transition(2,m,1). transition(2,n,1). transition(2,s,1). transition(2,t,1). transition(2,b,2). transition(2,e,2). transition(2,f,2). transition(2,h,2). transition(2,l,2). transition(2,r,2). transition(2,t,2). final(1).



%#program preference ( maxmin ).
%sum (P , S ) :- preference (P , maxmin ) , preference (P ,_ ,_ ,_ ,( S ,_ , _ )).



nd(S1,X,S2,S3) :- transition(S1,X,S2), transition(S1,X,S3), S2 != S3.

#show nd/4.
