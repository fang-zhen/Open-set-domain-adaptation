function [Cls,accO,accOs_star]=OSNNcv(Xs,Ys,Xt,Yt,T)
%number of Classes for Source domain
c=length(unique(Ys));
%number of Classes for target doamin
C=c+1;
%label begin
Cls=[];
%Compute the distance matrix
Dist=pdist2(Xs,Xt);
%dimension of features and number of samples
[mt,nt]=size(Xt);
%choice two cloest samples for target samples
[B,I]=mink(Dist,2,1);
%label every target sample
for i=1:mt
    %if two closet samples have same label
    if Ys(I(1,i))==Ys(I(2,i))
        Cls=[Cls;Ys(I(1,i))];
    else
    % Compare ratio of distances with threshold T
     if B(1,i)<=T*B(2,i)
         Cls=[Cls;Ys(I(1,i))];
     else if B(2,i)<=T*B(1,i)
             Cls=[Cls;Ys(I(2,i))];
         else 
             Cls=[Cls;c+1];
         end
     end
    end
end
        acc=0;
        %accuracy for AccOS
        for j=1:C
         L=find(Yt==j);
         acc=acc+length(find(Cls(L,1)==Yt(L,1)))/length(Yt(L,1));
        end
        accO=acc/C;
       
        
        %accuracy for AccOS*
        acc=0;
        for j=1:(C-1)
          L=find(Yt==j);
         acc=acc+length(find(Cls(L,1)==Yt(L,1)))/length(Yt(L,1));
        end
        accOs_star=acc/(C-1);
       
    
             
    
    