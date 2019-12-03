
clear;close all;

% Set algorithm parameters
options.p = 10;      %%%%manifold neighbour
   
options.sigma = 1;     %%%%regularization
    
options.lambda = 50; %%%%%MMD regula rization
    
options.rho = 1;   %%%%%mainfold regularization
   
options.T = 10;      %%%%iterations%%%%%
    
options.t = 0.50;    %%%%%%threshold t
    
options.alpha = 0.4 ;%%%%%%open set parameters 
    
options.gamma = 0.2    ;  %%%%%%open set parameters
   

srcStrVGG16 = {'b', 'b', 'b', 'i', 'i', 'i', 'c',  'c', 'c', 'p', 'p', 'p'};
tgtStrVGG16 = {'c', 'i', 'p', 'b', 'c', 'p', 'b',  'i', 'p', 'b',  'c', 'i'};
datafeature = 'VGG16';


results = []; 
for iData = 1:12

    if strcmp(datafeature,'VGG16')
        src = char(srcStrVGG16{iData});
        tgt = char(tgtStrVGG16{iData});
        options.data = strcat(src,'-vs-',tgt);
        fprintf('Data=%s \n',options.data);


        % load and preprocess data  
        load(['C:\Users\Fangzhen\Desktop\新建文件夹\DAOD  code\ImageCLEF_DA_VGG16\ImageDA_CLEF_VGG16\' src '.mat']);
        Xs = double((fea)); 
        Ys = double(labels');
        Zs=[Xs,Ys];
        Zs=sortrows(Zs,size(Zs,2));
        Xs=Zs(:,1:(size(Zs,2)-1));
        Ys=Zs(:,end)+1;
       

        load(['C:\Users\Fangzhen\Desktop\新建文件夹\DAOD  code\ImageCLEF_DA_VGG16\ImageDA_CLEF_VGG16\' tgt '.mat']);
        Xt = double((fea)); 
        Yt = double(labels');
        Zt=[Xt,Yt];
        Zt=sortrows(Zt,size(Zt,2));
        Xt=Zt(:,1:(size(Zt,2)-1));
        Yt=Zt(:,end)+1;
        
        %choice known classes and unknown classes:known classes 1-8,unknown classes 9-12
        
        [Xs,Xt,Ys,Yt]=datachoice(Xs,Xt,Ys,Yt,8,8);
    else
        break
    end
   Xs = Xs ./ repmat(sum(Xs,2),1,size(Xs,2));
   Xt = Xt ./ repmat(sum(Xt,2),1,size(Xt,2));
   Xs=normr(zscore(Xs)); 
   Xt=normr(zscore(Xt));
 
  [Acc_OS,Acc_OS_star,Yt_pred] = DAOD(Xs,Ys,Xt,Yt,options);  

  results = [results;[Acc_OS,Acc_OS_star]];
end
Average=sum(results)/12;
disp('Ave_Acc(OS)'); disp(Average(1,1));
disp('Ave_Acc(OS*)'); disp(Average(1,2));

