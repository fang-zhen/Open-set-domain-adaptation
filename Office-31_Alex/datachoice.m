function [Xs,Xt,Ys,Yt]=datachoice(X_src,X_tar,Y_src,Y_tar,c,C)
%%%shared classes 1 to c
%%%unknown classes C+1 to numclasses
s=find(Y_src<=c);
t0=find(Y_tar<=c);
t1=find(Y_tar>C);
t=[t0;t1];
Xs=X_src(s,:);
Ys=Y_src(s,1);
Xt=X_tar(t,:);
Yt=[Y_tar(t0,1);(c+1)*ones(length(t1),1)];
