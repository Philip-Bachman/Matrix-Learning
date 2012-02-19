function [ A ] = learn_sggm_bases(...
    X, basis_count, k, l1_lwr, l1_bases, step, round_count, Ai )
% Learn a set of sparse gaussian graphical model bases
%
% Parameters:
%   X: input observations for self-regression (obs_count x obs_dim)
%   basis_count: number of basis matrices to learn (scalar)
%   k: kernel width for locally-weighted regressions (scalar)
%   l1_lwr: l1 penalty to use during locally-weighted regressions (scalar)
%   l1_bases: l1 penalty to use for basis entries (scalar)
%   step: initial step size for gradient descent (scalar)
%   round_count: number of update rounds to perform (scalar)
%   Ai: (optional) set of starting bases (obs_dim x obs_dim x basis_count)
%

obs_count = size(X,1);
obs_dim = size(X,2);

if ~exist('Ai','var')
    A = rand_sggm_bases(obs_dim, basis_count);
else
    A = Ai(:,:,:);
end

do_cv = 0; % Whether to use hold-out set for line search in descent
nz_lvl = 0.1; % Noise level to add in gradient computations
% Learning loop
fprintf('Performing basis updates:\n');
for i=1:round_count,
    % Encode the input sequence using basis-projected sparse self-regression
    beta = lwr_matrix_sparse(X, X, A, k, l1_lwr, 0, 0);
    [ A_t post_err pre_err best_step ] = ...
        update_sggm_bases(A, beta, X, step, l1_bases, nz_lvl, do_cv);
    fprintf('    round: %d, pre_err: %.4f post_err: %.4f, step: %.4f, kurt: %.4f\n',...
        i, pre_err, post_err, best_step, kurtosis(A_t(:)));
    A = A_t(:,:,:);
    step = best_step * 1.1;
    nz_lvl = nz_lvl * 0.95;
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
