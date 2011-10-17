function [ A_new best_err pre_err best_step ] = update_bases(...
    A, B, X, Y, step_size, l1_pen, kill_diags, noise_lvl, do_cv )
% Update A along the gradient of ||y_out - sum_i(A_i*x_i*y_in)||^2 so as to
% decrease the error. We update the bases just a bit. Bases are updated with
% respect to all x, y_out, y_in in X, Y_out, Y_in respectively. That is, the
% gradient is averaged across these sets prior to basis updating.
%
% Parameters:
%   A: bases onto which inputs were projected prior to lwr. The bases can be
%      either 2D (for vector-scalar lwr) or 3D (for vector-vector lwr)
%   B: lwr coefficients for input/output pairs in X/Y (obs_count x basis_count)
%   X: the input observations (obs_count x in_dim)
%   Y: the output observations (obs_count x out_dim)
%   step_size: the amount to move along the gradient
%   l1_pen: L1 regularization penalty to add to basis gradient
%   kill_diags: Should we kill diagonal entries? 1=yes, 0=no.
%   noise_lvl: relative amount of noise to add to gradients
%   do_cv: whether to use a hold-out validation set for step size line search
%
% Output:
%   A_new: input bases, nudged along implied partial gradients
%   err: lwr error after doing basis update
%

% Set an L1 regularization term, to sparsify the learned basis matrices.
if ~exist('l1_pen', 'var'),
    l1_pen = 0.000;
end

if ~exist('kill_diags','var'),
    kill_diags = 0;
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
    train_idx = randsample(obs_count, round(4*obs_count/5));
else
    train_idx = 1:obs_count;
end
test_idx = setdiff(1:obs_count, train_idx);
in_dim = size(X,2);
out_dim = size(Y,2);
if (obs_count ~= size(Y,1))
    error('update_bases: Input/output observation count mismatch!\n');
end
if (in_dim ~= out_dim && out_dim ~= 1)
    error('update_bases: Only allows vector-vector or vector-scalar!\n');
end

% Average the gradients across all of the observations in X/Y
for i=1:numel(train_idx),
    idx = train_idx(i);
    b = B(idx,:)';
    x = X(idx,:)';
    y = Y(idx,:)';
    A_semi_grads = basis_gradients(A, b, x, y, l1_pen);
    A_grads = A_grads + A_semi_grads;
end
A_grads = A_grads ./ numel(train_idx);

% Add noise into the gradients, proportional to the gradient energy
if (noise_lvl > 0.01)
    A_grads = A_grads + (randn(size(A_grads)) .* (noise_lvl*std(A_grads(:))));
end

% Get the gradient descent step sizes to check in "line search"
trial_steps = linspace(0.1,1.0,5);
trial_steps = trial_steps .* step_size;

% Get the validation set with which to perform "line search"
if (do_cv == 1)
    Y_test = Y(test_idx,:);
    B_test = B(test_idx,:);
    X_test = X(test_idx,:);
else
    Y_test = Y(train_idx,:);
    B_test = B(train_idx,:);
    X_test = X(train_idx,:);
end

% Perform "line search" for some descent sizes, taking the best step at end
Y_hat = lwr_predict_matrix(X_test, A, B_test);
Yt_var = sum(sum((Y_test - repmat(mean(Y_test),size(Y_test,1),1)).^2));
pre_err = sum(sum((Y_test - Y_hat).^2)) / Yt_var;
%pre_err = pre_err + l1_pen * sum(abs(A(:)));
best_step = min(trial_steps);
best_err = pre_err;
for i=1:(numel(trial_steps)+1),
    if (i <= numel(trial_steps))
        step_size = trial_steps(i);
    else
        step_size = best_step;
    end
    % Move A down along the averaged gradient
    A_new = A - (A_grads .* step_size);
    % Kill diagonal entries if so desired
    if (kill_diags == 1)
        for j=1:size(A_new,3),
            for k=1:min(in_dim,out_dim),
                A_new(k,k,j) = 0;
            end
        end
    end
    % Normalize the bases to have unit variance (to constrain max eigenvalue)
    basis_count = size(A_new,3);
    for basis_num=1:basis_count,
        basis = A_new(:,:,basis_num);
        A_new(:,:,basis_num) = basis ./ std(basis(:));
    end
    % Compute the regression error after basis update with current step size
    Y_hat = lwr_predict_matrix(X_test, A_new, B_test);
    err = sum(sum((Y_test - Y_hat).^2)) / Yt_var;
    %err = err + l1_pen * sum(abs(A_new(:)));
    if ((err < best_err) || (i > numel(trial_steps)))
        % Note step sizes that improve on current best error
        best_err = err;
        best_step = step_size;
    end
end

return

end
