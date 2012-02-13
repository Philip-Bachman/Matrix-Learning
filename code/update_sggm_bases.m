function [ A_new best_err pre_err best_step ] = update_sggm_bases(...
    A, B, X, step_size, l1_pen, noise_lvl, do_cv )
% Update A along the gradient of ||y_out - sum_i(A_i*x_i*y_in)||^2 so as to
% decrease the error. We update the bases just a bit. Bases are updated with
% respect to all x, y_out, y_in in X, Y_out, Y_in respectively. That is, the
% gradient is averaged across these sets prior to basis updating.
%
% Parameters:
%   A: bases onto which inputs were projected prior to self regression
%   B: lwr coefficients for observations in X (obs_count x basis_count)
%   X: the input observations (obs_count x obs_dim)
%   step_size: the amount to move along the gradient
%   l1_pen: L1 regularization penalty to add to basis gradient
%   noise_lvl: relative amount of noise to add to gradients
%   do_cv: whether to use a hold-out validation set for step size line search
%
% Output:
%   A_new: input bases, nudged along implied partial gradients
%   best_err: lwr error after doing best-step basis update
%   pre_err: lwr error prior to basis update
%   best_step: step size of best update
%

basis_count = size(A,3);

% Set an L1 regularization term, to sparsify the learned basis matrices.
if ~exist('l1_pen', 'var'),
    l1_pen = 0.000;
end

if ~exist('noise_lvl','var'),
    noise_lvl = 0;
end

if ~exist('do_cv','var'),
    do_cv = 1;
end

A_grads = zeros(size(A));
obs_count = size(X,1);
if (do_cv == 1)
    train_idx = randsample(obs_count, round(3*obs_count/4));
else
    train_idx = 1:obs_count;
end
test_idx = setdiff(1:obs_count, train_idx);

% Average the gradients across all of the observations
for i=1:numel(train_idx),
    idx = train_idx(i);
    b = B(idx,:)';
    x = X(idx,:)';
    A_semi_grads = basis_gradients(A, b, x, x, l1_pen);
    A_grads = A_grads + A_semi_grads;
end
% Symmetrize the gradients (to account for use as sparse GGM bases)
for i=1:basis_count,
    basis = squeeze(A(:,:,i));
    A(:,:,i) = (basis + basis') ./ 2;
end
% Normalize basis gradients to account for number of training observations
A_grads = A_grads ./ numel(train_idx);

% Add symmetric noise into the gradients, proportional to the gradient energy
if (noise_lvl > 0.01)
    noise_scale = noise_lvl * std(A_grads(:));
    for i=1:basis_count,
        basis = squeeze(A_grads(:,:,i));
        noise = randn(size(A_grads,1),size(A_grads,2)) .* noise_scale;
        A_grads(:,:,i) = basis + noise + noise';
    end
end

% Get the gradient descent step sizes to check in "line search"
% trial_steps = linspace(0.1,1.0,5);
trial_steps = [0.1 0.5 1.0];
trial_steps = trial_steps .* step_size;

% Get the validation set with which to perform "line search"
if (do_cv == 1)
    X_test = X(test_idx,:);
    B_test = B(test_idx,:);
else
    X_test = X(train_idx,:);
    B_test = B(train_idx,:);
end

% Perform "line search" for some descent sizes, taking the best step at end
X_hat = lwr_predict_matrix(X_test, A, B_test);
Xt_var = sum(sum((bsxfun(@minus,X_test,mean(X_test))).^2));
pre_err = sum(sum((X_test - X_hat).^2)) / Xt_var;
%pre_err = pre_err + l1_pen * sum(abs(A(:)));
best_step = min(trial_steps);
best_err = pre_err;
for i=1:numel(trial_steps),
    step_size = trial_steps(i);
    % Move A down along the averaged gradient
    A_new = descent_update_sggm(A, A_grads, step_size);
    % Compute the regression error after basis update with current step size
    X_hat = lwr_predict_matrix(X_test, A_new, B_test);
    err = sum(sum((X_test - X_hat).^2)) / Xt_var;
    %err = err + l1_pen * sum(abs(A_new(:)));
    if ((err < best_err) || (i == 1))
        % Note step sizes that improve on current best error
        best_err = err;
        best_step = step_size;
    end
end

A_new = descent_update_sggm(A, A_grads, best_step);

return

end

function [ A_new ] = descent_update_sggm(A, A_grads, step_size)
% Do a descent update of A along the direction indicated by A_grads
basis_count = size(A,3);
diag_dim = min(size(A,1),size(A,2));
A_new = A - (A_grads .* step_size);
% Kill diagonal entries if so desired
for j=1:basis_count,
    for k=1:diag_dim,
        A_new(k,k,j) = 0;
    end
end
% Normalize the bases to have a standard deviation of 1
for basis_num=1:basis_count,
    basis = A_new(:,:,basis_num);
    A_new(:,:,basis_num) = basis ./ std2(basis);
end
return
end