
clear;close all;

% Set algorithm parameters
options.p = 10;      %%%%manifold neighbour
   
options.sigma = 1;     %%%%regularization
    
options.lambda = 500; %%%%%MMD regularization
    
options.rho = 1;     %%%%%mainfold regularization
   
options.T = 10;      %%%%iterations%%%%%in our paper, T=20
    
options.t = 0.50;    %%%%%%threshold t
    
options.alpha = 0.4; %%%%%%open set parameters
    
options.gamma = 0.25; %%%%%%open set parameters beta 
 

srcStrPIE = {'P1',   'P1',  'P1',  'P1', 'P2', 'P2', 'P2',  'P2',  'P3',  'P3',  'P3',  'P3', 'P4',  'P4',  'P4',  'P4', 'P5',  'P5',  'P5',  'P5'};
tgtStrPIE = {'P2',   'P3',  'P4',  'P5', 'P1', 'P3', 'P4',  'P5',  'P1',  'P2',  'P4',  'P5', 'P1',  'P2',  'P3',  'P5', 'P1',  'P2',  'P3',  'P4'};



results=[];

for iData = 1:20

    
        src = char(srcStrPIE{iData});
        tgt = char(tgtStrPIE{iData});
        options.data = strcat(src,'-vs-',tgt);
        fprintf('Data=%s \n',options.data);


        % load and preprocess data  
        load(['C:\Users\Fangzhen\Desktop\新建文件夹\DAOD  code\PIE\PIE_Data\' src 'data.mat']);
        Xs = fts; 
        Ys = labels;
       

        load(['C:\Users\Fangzhen\Desktop\新建文件夹\DAOD  code\PIE\PIE_Data\' tgt 'data.mat']);
        Xt = fts; 
        Yt = labels;
        
        %choice known classes and unknown classes:known classes 1-20,unknown classes 20-68
        
        [Xs,Xt,Ys,Yt]=datachoice(Xs,Xt,Ys,Yt,20,20);
        
        %for PIE, after normr and zscore, the performance is better
       
        Xs=zscore(normr(Xs));
        Xt=zscore(normr(Xt));
    
        [Acc_OS,Acc_OS_star,Yt_pred] = DAOD(Xs,Ys,Xt,Yt,options); 
        results = [results;[Acc_OS,Acc_OS_star]];
end
Average=sum(results)/20;
disp('Ave_Acc(OS)'); disp(Average(1,1));
disp('Ave_Acc(OS*)'); disp(Average(1,2));


 
