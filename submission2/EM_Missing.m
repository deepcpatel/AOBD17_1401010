% EM Algorithm
% Reference:- Probabilistic Principal Component Analysis by 
%             Michael E. Tipping; Christopher M. Bisho


function [W,sigma,M,mean,x_t] = EM_Missing(X,q,itr)

    x_row = size(X,1);
    x_col = size(X,2);          % N
    
    x_t = zeros(x_row,x_col);   % Mean of Probability of X given T
    
    mean = zeros(x_row,1);      % Mean mu
    
    W = randn(x_row,q);         % Initializing W 
    sigma =  randn(1);          % Initializing Sigma (Variance)
    
    for i = 1:1:x_col
        for j = 1:1:x_row
            if ~isnan(X(j,i))
                
                mean(j,1) = mean(j,1) + X(j,i);
            end
        end
    end
    
    mean = mean/x_col;
    
    S = zeros(x_row,x_row);              % Sample Covariance Matrix
    
    for i = 1:1:x_col 
        
        vec = X(:,i);
        
        for k = 1:1:x_row
            if isnan(vec(k,1))
                
                vec(k,1) = mean(k,1);
            end
        end
        
        product =(vec-mean)*(vec-mean)';
        
        S = S + product;
    end
    
    S = S/x_col;
    M = W'*W + sigma*eye(q);
    
    for i = 1:itr
       
        W_cap = S*W*inv(sigma*eye(q) + M\(W'*S*W));
        sigma_cap = trace(S - S*W*inv(M)*W_cap')/x_row;
        
        W = W_cap;
        sigma = sigma_cap;
        
        M = W'*W + sigma*eye(q);
        
    end
    
    for i = 1:x_col
        for k = 1:1:x_row
            if ~isnan(X(k,i))
                
                x_t(k,i) = (X(k,i)-mean(k,1));
            else
                x_t(k,i) = (X(k-1,i)-mean(k,1));     % Tweak
            end
        end
    end
    
    x_t = W'*(x_t); % Principle Components
    
end