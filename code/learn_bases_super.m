function [ A w ] = learn_bases_super(X, Y, opts)
% Learn a set of sparse gaussian graphical model bases, including a supervisory
% signal based on logistic regression using the linear combination of basis
% matrices selected for each observation, given some set of targets.
%
% Parameters:
%   X: input observations for self-regression (obs_count x obs_dim)
%   Y: target class for each inpupt observations (obs_count x 1)
%   opts: structure determining the following options:
%     basis_count: number of basis matrices to learn (scalar)
%     k: kernel width for locally-weighted regressions (scalar)
%     spars: desired sparsity for locally-weighted regressions (scalar)
%     l1_bases: l1 penalty to use for basis entries (scalar)
%     l_mix: mixing ratio (1 -> unsupervised only ... 0 -> supervised only)
%     step: initial step size for gradient descent (scalar)
%     round_count: number of update rounds to perform (scalar)
%     Ai: optional starting basis matrices (obs_dim x obs_dim x basis_count)
%     wi: optional starting classifier coefficients (basis_count x 1)
%     idx: optional indices into X/Y to use in updates
% Outputs:
%   A: learned basis matrices for self-regression
%   w: learned logistic regression coefficients
%
if ~exist('opts','var')
    error('learn_bases_super(): options structure required\n');
end
% Pull options out of options structure
basis_count = opts.basis_count;
k = opts.k;
spars = opts.spars;
round_count = opts.round_count;
if isfield(opts,'Ai')
    A = opts.Ai(:,:,:);
else
    A = rand_sggm_bases(size(X,2), basis_count);
end
if isfield(opts,'wi')
    w = opts.wi(:);
else
    w = zeros(basis_count, 1);
end
if isfield(opts,'idx')
    idx = opts.idx;
else
    idx = 1:size(X,1);
end

% Configure basis update options
%   up_opts:
%     step_size: the amount to move along the gradient
%     l1_bases: L1 regularization penalty to add to basis gradient
%     l2_reg: L2 regularizations used in elastic-net lwrs (obs_count x 1)
%     l_mix: mixing ratio (1 -> unsupervised only ... 0 -> supervised only)
%     noise_lvl: relative amount of noise to add to gradients
%     do_cv: whether to use hold-out validation set in step size line search
%     kill_diags: whether to kill diagonal basis entries during updates
up_opts = struct();
up_opts.step_size = opts.step;
up_opts.l1_bases = opts.l1_bases;
up_opts.l2_reg = zeros(length(idx),1);
up_opts.l_mix = opts.l_mix;
up_opts.noise_lvl = 0.1;
up_opts.do_cv = 0;
up_opts.kill_diags = 1;
% Learning loop
fprintf('Performing basis updates:\n');
Xtr = X(idx,:); Ytr = Y(idx);
for i=1:round_count,
    % Encode the input sequence using basis-projected sparse self-regression
    [B l2_reg] = lwr_matrix_sparse(X, X, A, k, spars, 0, 0, idx);
    up_opts.l2_reg = l2_reg;
    [A_t rec_e reg_e step] = update_bases_super(A, B, w, Xtr, Ytr, up_opts);
    fprintf('    round: %d, rec_err: %.4f reg_err: %.4f, step: %.4f, kurt: %.4f\n',...
        i, rec_e, reg_e, step, kurtosis(A_t(:)));
    A = A_t(:,:,:);
    up_opts.step_size = step * 1.25;
    up_opts.noise_lvl = up_opts.noise_lvl * 0.95;
end

return

end

function [ A ] = rand_sggm_bases(obs_dim, basis_count)
% Create a set of random sparse ggm bases (maybe not sparse)
%

A = randn(obs_dim, obs_dim, basis_count);
for i=1:basis_count,
    for j=1:obs_dim,
        A(j,j,i) = 0;
    end
    A(:,:,i) = squeeze(A(:,:,i)) + transpose(squeeze(A(:,:,i)));
    A(:,:,i) = A(:,:,i) ./ max(reshape(abs(A(:,:,i)),numel(A(:,:,i)),1));
end

return

end
