function [ A_hat ] = l1_reg_raw( X, sparsity, k, lwr_idx )
% Do either locally weighted l1-regularized self-regression, or plain
% l1-regularized self-regression on the sequence of observations in X.

in_dim = size(X,2);
MAX_STEP = round(3.0 * k);

% Setup the options structure for glmnet
options = glmnetSet();
options.dfmax = max(10,ceil(sparsity * in_dim));

if (lwr_idx > 0)
    % If an lwr is desired, set the range of observations to be used
    start_step = max(lwr_idx-MAX_STEP,1);
    end_step = min(lwr_idx+MAX_STEP,size(X,1));
    steps = start_step:end_step;
    % Compute normalized kernel weights for these observations
    weights = normpdf(steps-lwr_idx, 0, k);
    weights = weights ./ sum(weights);
    weights = diag(sqrt(weights));
    % Reweight the observations to be used in this lwr
    X_w = weights * X(steps,:);
else
    % Do not reweight observations if no lwr is desired
    X_w = X(:,:);
end

% Do an l1-regularized regression across the observations in X_w
A_hat = zeros(in_dim,in_dim);
for d=1:in_dim,
    Xd = X_w(:,:);
    % Kill this dimension, to zero diagonals in the learned matrix
    Xd(:,d) = 0;
    if (sparsity < 0.99)
        % Use L1-regularized regression if a sparse fit is desired
        fit = glmnet(Xd, X_w(:,d), 'gaussian', options);
        A_hat(d,:) = fit.beta(:,end)';
        for j=2:numel(fit.lambda),
            if ((fit.df(j) / in_dim) > sparsity)
                A_hat(d,:) = fit.beta(:,j-1)';
                break
            end
        end
    else
        % Do a regular linear regression if a dense fit is desired
        A_hat(d,:) = (pinv(Xd) * X_w(:,d))';
    end
end

return

end

