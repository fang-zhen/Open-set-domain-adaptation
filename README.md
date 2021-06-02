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

 6. If using the code, please cite paper:
 Open set domain adaptation: theoretical bound and algorithms. In TNNLS

@ARTICLE{9186366,
  author={Fang, Zhen and Lu, Jie and Liu, Feng and Xuan, Junyu and Zhang, Guangquan},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={Open Set Domain Adaptation: Theoretical Bound and Algorithm}, 
  year={2020},
  volume={},
  number={},
  pages={1-14},
  doi={10.1109/TNNLS.2020.3017213}}
