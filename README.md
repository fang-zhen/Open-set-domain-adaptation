1.  Implement_XXXX is the main text. All parameters have been tuned and fixed. One can implement code directly after changing the path.

2. The features of Office-31, Office-Home and Imageclef-DA are extracted from deep models (pretrained without finetune).

3. For all VGG features, we use the following data preprocessing method:

   Xs = Xs ./ repmat(sum(Xs,2),1,size(Xs,2));

   Xt = Xt ./ repmat(sum(Xt,2),1,size(Xt,2));

   Xs=normr(zscore(Xs)); 

   Xt=normr(zscore(Xt));
   
   The preprocessing method follows JGSA https://documents.uow.edu.au/~jz960/.
   
 4. The original code is written using Matlab R2019a. 
 
 5. There are 98 open set domain adaptation tasks.

