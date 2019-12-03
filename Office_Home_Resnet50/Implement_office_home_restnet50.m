
clear;close all;

% Set algorithm parameters
options.p = 10;      %%%%manifold neighbour
   
options.sigma = 1;     %%%%regularization
    
options.lambda = 500; %%%%%MMD regularization
    
options.rho = 1;   %%%%%mainfold regularization
   
options.T = 10;      %%%%iterations%%%%%
    
options.t = 0.50;    %%%%%%threshold t
    
options.alpha = 0.4;%%%%%%open set parameters
    
options.gamma = 0.25;  %%%%%%open set parameters
  

srcStrResNet50 = {'Ar',   'Ar',  'Ar',  'Cl', 'Cl', 'Cl', 'Pr',  'Pr',  'Pr',  'Rw',  'Rw',  'Rw'};
tgtStrResNet50 = {'Cl',   'Pr',  'Rw',  'Ar', 'Pr', 'Rw', 'Ar',  'Cl',  'Rw',  'Ar',  'Pr',  'Cl'};
datafeature = 'ResNet50';


results = []; 
for iData = 1:12

    if strcmp(datafeature,'ResNet50')
        src = char(srcStrResNet50{iData});
        tgt = char(tgtStrResNet50{iData});
        options.data = strcat(src,'-vs-',tgt);
        fprintf('Data=%s \n',options.data);


        % load and preprocess data  
        load(['C:\Users\Fangzhen\Desktop\新建文件夹\DAOD  code\Office_Home_Resnet50\Office-Home\' src '_resnet50.mat']);
        Xs = fts; 
        Ys = labels;
       

        load(['C:\Users\Fangzhen\Desktop\新建文件夹\DAOD  code\Office_Home_Resnet50\Office-Home\' tgt '_resnet50.mat']);
        Xt = fts; 
        Yt = labels;
        
        %choice known classes and unknown classes:known classes 1-25,unknown classes 26-65
        
        [Xs,Xt,Ys,Yt]=datachoice(Xs,Xt,Ys,Yt,25,25);
    else
        break
    end
    [Acc_OS,Acc_OS_star,Yt_pred] = DAOD(Xs,Ys,Xt,Yt,options);  

    results = [results;[Acc_OS,Acc_OS_star]];
end
Average=sum(results)/12;
disp('Ave_Acc(OS)'); disp(Average(1,1));
disp('Ave_Acc(OS*)'); disp(Average(1,2));

