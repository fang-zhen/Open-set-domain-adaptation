function  [Acc_OS,Acc_OS_star,Yt_pred] = DAOD(Xs,Ys,Xt,Yt,options)

%% Inputs:
%%% Xs      : Source domain feature matrix, n * dim
%%% Ys      : Source domain label matrix, n * 1
%%% Xt      : Target domain feature matrix, m * dim
%%% Yt      : Target domain label matrix, m * 1 (only used for testing accuracy)
%%% options : algorithm options:


%% Outputs:
%%%% Acc_OS      :  Final accuracy value
%%%% Acc_OS_star :  Final accuracy value
%%%% Beta        :  Cofficient matrix
%%%% Yt_pred     :  Prediction labels for target domain

%% Algorithm starts here
    fprintf('DAOD starts...\n');
    
    %% Load algorithm options
    if ~isfield(options,'p')
        options.p = 10;
    end
    if ~isfield(options,'sigma')
        options.eta = 1;
    end
    if ~isfield(options,'lambda')
        options.lambda = 50;
    end
    if ~isfield(options,'rho')
        options.rho = 1.0;
    end
    if ~isfield(options,'T')
        options.T = 10;
    end
    if ~isfield(options,'t')
        options.t = 0.5;
    end
    if ~isfield(options,'alpha')
        options.alpha = 0.40;
    end
    if ~isfield(options,'gamma')
        options.beta = 0.35;
    end
    
    Xs = double(Xs');
    Xt = double(Xt');

    X = [Xs,Xt];
    n = size(Xs,2);
    m = size(Xt,2);
    C = length(unique(Ys));
    acc_iter = [];
    
    YY = [];
    for c = 1 : (C+1)
        YY = [YY,Ys==c];
    end
    YY = [YY;[zeros(m,C),ones(m,1)]];
    YY2= [[zeros(n,C),ones(n,1)];zeros(m,C+1)];

    %% Data normalization
    X = X * diag(sparse(1 ./ sqrt(sum(X.^2))));

    %% Construct graph Laplacian
    if options.rho > 0
        manifold.k = options.p;
        manifold.Metric = 'Cosine';
        manifold.NeighborMode = 'KNN';
        manifold.WeightMode = 'Cosine';
        W = lapgraph(X',manifold);
        Dw = diag(sparse(sqrt(1 ./ sum(W))));
        L = eye(n + m) - Dw * W * Dw;
    else
        L = 0;
    end
    
    % Generate soft labels for the target domain
  
    Cls =OSNNcv(Xs',Ys,Xt',Yt,options.t);
    
    Cls_check_convergence=Cls;
    

    % Construct kernel
    K = kernel_meda('rbf',X,sqrt(sum(sum(X .^ 2).^0.5)/(n + m)));
    tl = max(m,n);
    
    E = diag(sparse([tl*ones(n,1)/n;options.alpha*tl*ones(m,1)/m]));
    E2 = diag(sparse([options.gamma*tl*ones(n,1)/n;zeros(m,1)/m]));

    for t = 1 : options.T
        
        % Estimate mu
        known_position = find(Cls<(C+1));
        Xt_known = Xt';
        Xt_known = Xt_known(known_position,:);
        Cls_known= Cls(known_position,1);
        % Estimate mu
        mu = estimate_mu(Xs',Ys,Xt_known,Cls_known);
        % Construct MMD matrix
        LL=zeros(m,1);
        LL(known_position,1)=1;
        k = sum(LL);
        e = [1 / n * ones(n,1); -1 / k * LL];
        M = e * e' * length(unique(Ys));
        N = 0;
        for c = reshape(unique(Ys),1,length(unique(Ys)))
            e = zeros(n + m,1);
            e(Ys == c) = 1 / length(find(Ys == c));
            e(n + find(Cls == c)) = -1 / length(find(Cls == c));
            e(isinf(e)) = 0;
            N = N + e * e';
        end
        M = (1 - mu) * M + mu * N;
        M = M / norm(M,'fro');

        % Compute coefficients vector Beta
        Beta = ((E-E2 + options.lambda * M + options.rho * L) * K + options.sigma * speye(n + m,n + m)) \ (E * YY-E2*YY2);
        F = K * Beta;
        [~,Cls] = max(F,[],2);
         Cls = Cls(n+1:end);
        %% Compute accuracy
       if Cls== Cls_check_convergence
           break;
       else
           Cls_check_convergence=Cls;
       end
        
    end
       Yt_pred = Cls;
       Acc_OS=0;
       Acc_OS_star=0;
         
       for j=1:(C+1)
         LL=find(Yt==j);
         Acc_OS=Acc_OS+length(find(Cls(LL,1)==Yt(LL,1)))/length(Yt(LL,1));
       end
         Acc_OS=Acc_OS/(C+1);
         
       
       for j=1:C
         LL=find(Yt==j);
         Acc_OS_star=Acc_OS_star+length(find(Cls(LL,1)==Yt(LL,1)))/length(Yt(LL,1));
       end
        Acc_OS_star=Acc_OS_star/C;
        disp('Acc(OS)');  disp(Acc_OS);
        disp('Acc(OS*)'); disp(Acc_OS_star);
    fprintf('DAOD ends!\n');
end

function K = kernel_meda(ker,X,sigma)
    switch ker
        case 'linear'
            K = X' * X;
        case 'rbf'
            n1sq = sum(X.^2,1);
            n1 = size(X,2);
            D = (ones(n1,1)*n1sq)' + ones(n1,1)*n1sq -2*X'*X;
            K = exp(-D/(2*sigma^2));        
        case 'sam'
            D = X'*X;
            K = exp(-acos(D).^2/(2*sigma^2));
        otherwise
            error(['Unsupported kernel ' ker])
    end
end
