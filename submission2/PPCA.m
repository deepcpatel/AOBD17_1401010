% PPCA
% Reference:- Probabilistic Principal Component Analysis by 
%             Michael E. Tipping; Christopher M. Bishop

function [W,sigma,M,mean,x_t] = PPCA(X,q)

    x_row = size(X,1);
    x_col = size(X,2);      % N
    
    x_t = zeros(x_row,x_col);   % Mean of Probability of X given T
    
    mean = zeros(x_row,1);  % Mean mu
    
    for i = 1:1:x_col
       
        mean = mean + X(:,i);
        
    end
    
    mean = mean/x_col;
    
    S = 0;              % Sample Covariance Matrix
    
    for i = 1:1:x_col
       
        S = S + ((X(:,i)-mean)*(X(:,i)-mean)');
        
    end
    
    S = S/x_col;
    
    [U,eval] = eig(S);   % Eigen Decomposition
    
    eval = diag(eval);
    [eval, i] = sort(eval, 'descend');
    U = U(:,i);
    eval = diag(eval);
    
    U = U(:,1:q);
    lambda = eval(1:q,1:q);
    
    if q == size(eval,2)
        
        sigma = 0;
    else
        
        sigma = 0;
        for i = q+1:1:size(eval,2)

            sigma = sigma + eval(i,i);

        end
    end
    
    sigma = sigma/(size(eval,2)-q); % Variance
    
    W = U*((lambda - sigma*eye(q)).^(1/2));
    
    M = W'*W + sigma*eye(q);
    
    for i = 1:x_col
       
        x_t(:,i) = (X(:,i)-mean);
        
    end
    
    x_t = (M\W')*(x_t);      % M\W stands for inv(M)*W in MATLAB, 
                             % which is faster and efficient than the later one
end