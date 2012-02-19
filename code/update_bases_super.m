function [ A_new rec_err reg_err best_step ] = update_bases_super(...
    A, B, w, X, Y, opts )
% Update A along the gradient of ||y_out - sum_i(A_i*x_i*y_in)||^2 so as to
% decrease the error. We update the bases just a bit. Bases are updated with
% respect to all x, y_out, y_in in X, Y_out, Y_in respectively. That is, the
% gradient is averaged across these sets prior to basis updating.
%
% Parameters:
%   A: bases onto which inputs were projected prior to self regression
%   B: lwr coefficients for observations in X (obs_count x basis_count)
%   w: logistic regression coefficients (basis_count x 1)
%   X: the input observations (obs_count x obs_dim)
%   Y: target class for each observation in X (obs_count x 1)
%   opts: structure determining the following options:
%     step_size: the amount to move along the gradient
%     l1_bases: L1 regularization penalty to add to basis gradient
%     l2_reg: L2 regularizations used in elastic-net lwrs (obs_count x 1)
%     l_mix: mixing ratio (1 -> unsupervised only ... 0 -> supervised only)
%     noise_lvl: relative amount of noise to add to gradients
%     do_cv: whether to use hold-out validation set in step size line search
%     kill_diags: whether to kill diagonal basis entries during updates
%
% Output:
%   A_new: input bases, nudged along implied partial gradients
%   rec_err: reconstruction error after doing best-step basis update
%   reg_err: l1 regularization error after basis update
%   best_step: step size of best update
%
basis_count = size(A,3);
A_grads = zeros(size(A));
obs_count = size(X,1);
if ~exist('opts','var')
    error('update_bases_super(): options structure required\n');
end
% Unpack options structure
step_size = opts.step_size;
l1_bases = opts.l1_bases;
l2_reg = opts.l2_reg;
if (length(l2_reg) == 1)
    l2_reg = ones(obs_count,1) .* l2_reg;
end
l_mix = opts.l_mix;
if isfield(opts,'noise_lvl'),
    noise_lvl = opts.noise_lvl;
else
    noise_lvl = 0;
end
if isfield(opts,'do_cv'),
    do_cv = opts.do_cv;
else
    do_cv = 1;
end
if isfield(opts,'kill_diags')
    kill_diags = opts.kill_diags;
else
    kill_diags = 1;
end
% Set up a hold-out validation set if desired
if (do_cv == 1)
    train_idx = randsample(obs_count, round(3*obs_count/4));
else
    train_idx = 1:obs_count;
end
test_idx = setdiff(1:obs_count, train_idx);
% Average the gradients across observations not in the test set
for i=1:numel(train_idx),
    idx = train_idx(i);
    b = B(idx,:)';
    x = X(idx,:)';
    y = Y(idx);
    A_semi_grads = basis_gradients_super(...
        A, b, w, x, x, y, l1_bases, l2_reg(idx), l_mix);
    A_grads = A_grads + A_semi_grads;
end
% Symmetrize the bases and gradients (to account for use as GGM structures)
for i=1:basis_count,
    basis = squeeze(A(:,:,i));
    A(:,:,i) = (basis + basis') ./ 2;
    grad = squeeze(A_grads(:,:,i));
    A_grads(:,:,i) = (grad + grad') ./ 2;
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
trial_steps = [0.1 0.25 0.5 1.0];
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
test_size = size(X_test,1);
X_hat = lwr_predict_matrix(X_test, A, B_test);
rec_err = sum(sum((X_test - X_hat).^2)) / test_size;
reg_err = l1_bases * sum(sum(abs(A(:))));
best_step = min(trial_steps);
for i=1:numel(trial_steps),
    step_size = trial_steps(i);
    % Move A down along the averaged gradient
    A_new = descent_update_sggm(A, A_grads, step_size, kill_diags);
    % Compute the errors after basis update with current step size
    X_hat = lwr_predict_matrix(X_test, A_new, B_test);
    err_1 = sum(sum((X_test - X_hat).^2)) / test_size;
    err_2 = l1_bases * sum(abs(A_new(:)));
    if (((err_1 + err_2) < (rec_err + reg_err)) || (i == 1))
        % Note step sizes that improve on current best error
        rec_err = err_1;
        reg_err = err_2;
        best_step = step_size;
    end
end
A_new = descent_update_sggm(A, A_grads, best_step, kill_diags);
return

end

function [ A_new ] = descent_update_sggm(A, A_grads, step, kill_diags)
% Do a descent update of A along the direction indicated by A_grads
basis_count = size(A,3);
diag_dim = min(size(A,1),size(A,2));
A_new = A - (A_grads .* step);
% Kill diagonal entries if so desired
if (kill_diags == 1)
    for j=1:basis_count,
        for k=1:diag_dim,
            A_new(k,k,j) = 0;
        end
    end
end
% Normalize the bases to have a standard deviation of 1
for basis_num=1:basis_count,
    basis = A_new(:,:,basis_num);
    A_new(:,:,basis_num) = basis ./ std2(basis);
end
return
end