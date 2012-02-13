function [ sigma ] = rand_sparse_cov( dim, spars, min_eig, do_prec )
% Generate a random sparse covariance matrix, based on a random correlation
% matrix, in which some fraction of entries have been (symmetrically) forced to
% zero at random.
%
% Parameters:
%   dim: dimension of the dim x dim matrix to generate
%   spars: the sparsity target rate (lesser is more sparse)
%   min_eig: minimum eigenvalue constraint, achieved by diagonal expansion
%   do_prec: consider the generated matrix as a precision matrix, and return
%            pseudoinverse to be used as a covariance matrix
%
% Output:
%   sigma: a somewhat random sparse covariance matrix of size dim x dim
%

if ~exist('do_prec','var')
    do_prec = 0;
end

% Turn off warnings because of weird warnings from eigs()
warning off all;

sigma = gallery('randcorr',dim);
keep_count = ceil(((dim * dim) - dim) * spars);
sigma_zd = sigma - diag(diag(sigma));
[corrs_sorted sort_idx] = sort(abs(sigma_zd(:)),'descend');
min_keep_corr = corrs_sorted(keep_count)-0.0001;
sigma(abs(sigma) < min_keep_corr) = 0;

eig_vals = eigs(sigma, dim);
I_step = eye(dim) .* 0.01;
iters = 0;

while (min(eig_vals) < min_eig)
    sigma = sigma + I_step;
    eig_vals = eigs(sigma, dim);
    iters = iters + 1;
end

if (do_prec == 1)
    sigma = pinv(sigma);
    % Forcefully symmetrize, to compensate for numerical slippage in pinv()
    sigma = (sigma + sigma') ./ 2;
end

sigma = sigma ./ mean(sqrt(diag(sigma)));

return

end

