
clear;close all;

% Set algorithm parameters
options.p = 10;      %%%%manifold neighbour
   
options.sigma = 1;     %%%%regularization
    
options.lambda = 50; %%%%%MMD regularization
    
options.rho = 1;   %%%%%mainfold regularization
   
options.T = 10;      %%%%iterations%%%%%
    
options.t = 0.50;    %%%%%%threshold t
    
options.alpha = 0.4 ;%%%%%%open set parameters 
    
options.gamma = 0.2  ;  %%%%%%open set parameters
   

srcStrResNet50 = {'b', 'b', 'b', 'i', 'i', 'i', 'c',  'c', 'c', 'p', 'p', 'p'};
tgtStrResNet50 = {'c', 'i', 'p', 'b', 'c', 'p', 'b',  'i', 'p', 'b',  'c', 'i'};
datafeature = 'ResNet50';


results = []; 
for iData = 1:12

    if strcmp(datafeature,'ResNet50')
        src = char(srcStrResNet50{iData});
        tgt = char(tgtStrResNet50{iData});
        options.data = strcat(src,'-vs-',tgt);
        fprintf('Data=%s \n',options.data);


        % load and preprocess data  
        load(['C:\Users\Fangzhen\Desktop\新建文件夹\DAOD  code\ImageCLEF_DA_ResNet50\ImageDA_CLEF_ResNet50\' src '.mat']);
        Xs = double((fea)); 
        Ys = double(labels')+1;
        Zs=[Xs,Ys];
        Zs=sortrows(Zs,size(Zs,2));
        Xs=Zs(:,1:(size(Zs,2)-1));
        Ys=Zs(:,end);
       

        load(['C:\Users\Fangzhen\Desktop\新建文件夹\DAOD  code\ImageCLEF_DA_ResNet50\ImageDA_CLEF_ResNet50\' tgt '.mat']);
        Xt = double((fea)); 
        Yt = double(labels')+1;
        Zt=[Xt,Yt];
        Zt=sortrows(Zt,size(Zt,2));
        Xt=Zt(:,1:(size(Zt,2)-1));
        Yt=Zt(:,end);
        
        %choice known classes and unknown classes:known classes 1-8,unknown classes 8-12
        
        [Xs,Xt,Ys,Yt]=datachoice(Xs,Xt,Ys,Yt,8,8);
    else
        break
    end
   
 
   [Acc_OS,Acc_OS_star,Yt_pred] = DAOD(Xs,Ys,Xt,Yt,options);  

   results = [results;[Acc_OS,Acc_OS_star]];
end
Average=sum(results)/12;
disp('Ave_Acc(OS)'); disp(Average(1,1));
disp('Ave_Acc(OS*)'); disp(Average(1,2));

