
clear;close all;

% Set algorithm parameters
options.p = 10;      %%%%manifold neighbour
   
options.sigma = 1;     %%%%regularization
    
options.lambda = 500   ; %%%%%MMD regularization
    
options.rho = 1;   %%%%%mainfold regularization
   
options.T = 10;      %%%%iterations%%%%%
    
options.t = 0.50;    %%%%%%threshold t
    
options.alpha = 0.4 ;%%%%%%open set parameters 
    
options.gamma = 0.15 ;  %%%%%%open set parameters
   

srcStrVGG19 = {'Ar',   'Ar',  'Ar',  'Cl', 'Cl', 'Cl', 'Pr',  'Pr',  'Pr',  'Rw',  'Rw',  'Rw'};
tgtStrVGG19 = {'Cl',   'Pr',  'Rw',  'Ar', 'Pr', 'Rw', 'Ar',  'Cl',  'Rw',  'Ar',  'Pr',  'Cl'};
datafeature = 'VGG19';
results = []; 
for iData = 1:12

    if strcmp(datafeature,'VGG19')
        src = char(srcStrVGG19{iData});
        tgt = char(tgtStrVGG19{iData});
        options.data = strcat(src,'-vs-',tgt);
        fprintf('Data=%s \n',options.data);


        % load and preprocess data  
        load(['C:\Users\Fangzhen\Desktop\新建文件夹\DAOD  code\Office_Home_VGG19\Office-Home\' src '_VGG19.mat']);
        Xs = double((fea)); 
        Ys = double(labels');
        Zs=[Xs,Ys];
        Zs=sortrows(Zs,size(Zs,2));
        Xs=Zs(:,1:(size(Zs,2)-1));
        Ys=Zs(:,end)+1;
       

        load(['C:\Users\Fangzhen\Desktop\新建文件夹\DAOD  code\Office_Home_VGG19\Office-Home\' tgt '_VGG19.mat']);
        Xt = double((fea)); 
        Yt = double(labels')+1;
        Zt=[Xt,Yt];
        Zt=sortrows(Zt,size(Zt,2));
        Xt=Zt(:,1:(size(Zt,2)-1));
        Yt=Zt(:,end);
        
        %choice known classes and unknown classes:known classes 1-25,unknown classes 26-65
        
        [Xs,Xt,Ys,Yt]=datachoice(Xs,Xt,Ys,Yt,25,25);
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

