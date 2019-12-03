1. One can implement code after changing the path. Implementation_XXXX is the main text.

2. The features of Office-31, Office-Home and Imageclef-DA are extracted from deep models (pretrained without finetune).

3. For all VGG features, we use the following data preprocessing method:
Xs = Xs ./ repmat(sum(Xs,2),1,size(Xs,2));
Xt = Xt ./ repmat(sum(Xt,2),1,size(Xt,2));
Xs=normr(zscore(Xs)); 
Xt=normr(zscore(Xt));
The preprocessing method was learned from JGSA https://documents.uow.edu.au/~jz960/.

4. We will update our code and provide an official version after paper is accepted.
