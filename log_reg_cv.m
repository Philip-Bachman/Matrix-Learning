function [ cv_err mean_beta ] = log_reg_cv( X, Y, rounds, sparsity, test_frac, add_bias )
% Do randomized cross-validation on the given input X/Y, with X being the
% features and Y the class labels. 
%
% Parameters:
%   X: input observations (obs_count x in_dim)
%   Y: target class labels (obs_count x 1) (label in {1, 2})
%   rounds: number of cv rounds to perform
%   sparsity: target number of non-zero coefficients
%   test_frac: proportion of examples to put in test set
%   add_bias: whether or not to add a bias term
%
% Output:
%   cv_err: classification error in each round of cross-validation
%

obs_count = size(X,1);

if (size(Y,1) ~= obs_count)
    error('log_reg_cv: mismatched input/output observation counts!\n');
end

if ~exist('add_bias','var')
    add_bias = 0;
end

if (add_bias == 1)
    mean_beta = zeros(size(X,2)+1,1);
else
    mean_beta = zeros(size(X,2),1);
end

cv_err = zeros(1,rounds);
test_size = round(obs_count * test_frac);

fprintf('Doing %d rounds of lr-cv:',rounds);
for r_num=1:rounds,
    if (mod(r_num, ceil(rounds/60)) == 0)
        fprintf('.');
    end
    % Set up the training and test sets for this round of CV
    idx = randperm(obs_count);
    test_idx = idx(1:test_size);
    train_idx = idx(test_size+1:end);
    X_test = X(test_idx,:);
    Y_test = Y(test_idx,1);
    X_train = X(train_idx,:);
    Y_train = Y(train_idx,1);
    % Learn a sparse set of logistic regression coefficients
    beta = l1_log_reg(X_train, Y_train, sparsity, add_bias);
    mean_beta = mean_beta + beta;
    % Append a bias if one was used in coefficient learning
    if (add_bias == 1);
        X_test = [X_test ones(test_size,1)];
    end
    % Test the coefficients on the test set
    Y_pred = X_test * beta;
    c1 = Y_pred < 0;
    Y_pred(c1) = 1;
    Y_pred(~c1) = 2;
    miss_count = sum(Y_pred ~= Y_test);
    cv_err(r_num) = miss_count / numel(Y_test);
end
fprintf('\n');
mean_beta = mean_beta ./ rounds;

return

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function for learning a logistic-regression classifier
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ beta ] = l1_log_reg( X, Y, sparsity, add_bias )
% Use glmnet to do L1-regularized logistic regression using the inputs in X and
% outputs in Y, with an eye on getting a fraction sparsity non-zeros
%
% Parameters:
%   X: input observations (obs_count x obs_dim)
%   Y: output binary classes (obs_count x 1)
%   sparsity: desired fraction of non-zero regression coefficients
%   add_bias: whether or not to add a bias term
%
% Output:
%   beta: a vector of regression coefficients (obs_dim x 1)
%

if ~exist('add_bias','var')
    add_bias = 0;
end

if (add_bias == 1)
    X = [X ones(size(X,1),1)];
end

obs_dim = size(X,2);

% Set options for glmnet
options = glmnetSet();
options.penalty_factor = ones(1,obs_dim);
options.dfmax = max(10,ceil(sparsity * obs_dim));
options.maxit = 200;
options.alpha = 0.75;
if (size(X,2) < 5)
    options.lambda_min = 0.01;
end

if (add_bias == 1)
    options.penalty_factor(obs_dim) = 0;
end

fit = glmnet(X, Y, 'binomial', options);
beta = fit.beta(:,end);
for i=2:numel(fit.lambda),
    if ((fit.df(i) / obs_dim) > sparsity)
        beta = fit.beta(:,i-1);
        break
    end
end

return

end