% EM Algorithm
% Reference:- Probabilistic Principal Component Analysis by 
%             Michael E. Tipping; Christopher M. Bisho


function [W,sigma,M,mean,x_t] = EM(X,q,itr)

    x_row = size(X,1);
    x_col = size(X,2);          % N
    
    x_t = zeros(x_row,x_col);   % Mean of Probability of X given T
    
    mean = zeros(x_row,1);      % Mean mu
    
    W = randn(x_row,q);         % Initializing W 
    sigma =  randn(1);          % Initializing Sigma (Variance)
    
    for i = 1:1:x_col
       
        mean = mean + X(:,i);
        
    end
    
    mean = mean/x_col;
    
    S = 0;              % Sample Covariance Matrix
    
    for i = 1:1:x_col
       
        S = S + ((X(:,i)-mean)*(X(:,i)-mean)');
        
    end
    
    S = S/x_col;
    
    for i = 1:itr
        
        M = W'*W + sigma*eye(q);
        
        mult = S*W;
        inv_M = inv(M);
        
        W_cap = mult*inv(sigma*eye(q) + inv_M*(W'*mult));
        sigma_cap = trace(S - mult*inv_M*W_cap')/x_row;
        
        W = W_cap;
        sigma = sigma_cap;
    end
    
    for i = 1:x_col
       
        x_t(:,i) = (X(:,i)-mean);
        
    end
    
    x_t = W'*(x_t); % Principle Components
    
end