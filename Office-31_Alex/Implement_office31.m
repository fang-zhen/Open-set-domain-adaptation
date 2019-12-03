
clear;close all;

% Set algorithm parameters
options.p = 10;      %%%%manifold neighbour
   
options.sigma = 1;     %%%%regularization
    
options.lambda = 50; %%%%%MMD regularization
    
options.rho = 1;   %%%%%mainfold regularization
   
options.T = 10;      %%%%iterations%%%%%
    
options.t = 0.50;    %%%%%%threshold t
    
options.alpha = 0.4;%%%%%%open set parameters
    
options.gamma = 0.2; %%%%%%open set parameters
 

srcStrAlexnet7 = {'amazon',   'amazon',  'webcam',  'webcam', 'dslr',   'dslr'};
tgtStrAlexnet7 = {'webcam',   'dslr',    'amazon',  'dslr',   'amazon', 'webcam'};




datafeature = 'Alexnet7';




results = [];
for iData = 1:6

    if strcmp(datafeature,'Alexnet7')
        src = char(srcStrAlexnet7{iData});
        tgt = char(tgtStrAlexnet7{iData});
        options.data = strcat(src,'-vs-',tgt);
        fprintf('Data=%s \n',options.data);


        % load and preprocess data  
        load(['C:\Users\Fangzhen\Desktop\新建文件夹\DAOD  code\Office-31_Alex\Data_office31\' src '_Al7.mat']);
        Xs = fts; 
        Ys = labels;
       

        load(['C:\Users\Fangzhen\Desktop\新建文件夹\DAOD  code\Office-31_Alex\Data_office31\' tgt '_Al7.mat']);
        Xt = fts; 
        Yt = labels;
        
        %choice known classes and unknown classes:known classes 1-10,unknown classes 21-31
        [Xs,Xt,Ys,Yt]=datachoice(Xs,Xt,Ys,Yt,10,20);
    else
        break
    end
    [Acc_OS,Acc_OS_star,Yt_pred] = DAOD(Xs,Ys,Xt,Yt,options);
    
    
    results = [results;[Acc_OS,Acc_OS_star]];
end
Average=sum(results)/6;
disp('Ave_Acc(OS)'); disp(Average(1,1));
disp('Ave_Acc(OS*)'); disp(Average(1,2));

