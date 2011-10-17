function [ ppc_bases mean_basis l_sums ] = learn_matrix_bases_ppc( X, Y, k, sparsity, clip, drop, do_mean, rp_count, idx, kill_diags )
% Learn set of PPC bases for the given input/output observation sequences. Use
% L1-regularized least-squares regression during the initial lwr phase.
%
% Parameters:
%   X: input observation sequence (obs_count x in_dim)
%   Y: output observation sequence (obs_count x out_dim)
%   k: Gaussian kernel width to use when learning bases
%   sparsity: the max allowed density of the lwr coefficients during learning
%   clip: ? include only observations up to the current point
%   drop: ? drop the current observation during lwr
%   do_mean: ? compute a mean basis after which lwr is done on residuals
%   rp_count: number of random projections to use for PPC approximation
%   idx: list of indices into X/Y at which to perform lwr
%   kill_diags: ? kill diagonal values in basis matrices
%
% Output:
%   ppc_bases: the bases learned via Parameter Principal Components
%   mean_basis: the mean basis, or a matrix of zeros
%   l_sums: cumulative sums of latent over the Parameter Principal Components
%

if ~exist('clip','var')
    clip = 0;
end
if ~exist('drop','var')
    drop = 0;
end
if ~exist('do_mean','var')
    do_mean = 0;
end
if ~exist('rp_count','var')
    rp_count = 10;
end
if ~exist('idx','var')
    idx = 1:size(X,1);
end
if ~exist('kill_diags','var')
    kill_diags = 0;
end


obs_count = size(X,1);
in_dim = size(X,2);
out_dim = size(Y,2);

% Check for copacetic parameter dimensions
if (size(Y,1) ~= obs_count)
    error('learn_matrix_bases_sparse: mismatched input/output obs counts!\n');
end
if (in_dim ~= out_dim)
    error('learn_matrix_bases_sparse: only square basis matrices allowed!\n');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Do l1-regularized regression to compute the mean basis
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Setup the options structure for glmnet
options = glmnetSet();
options.dfmax = max(10,ceil(sparsity * in_dim));

if (do_mean == 1)
    fprintf('Learning mean matrix:');
    % Do regression on all observations to get mean coefficients
    mean_basis = zeros(in_dim,in_dim);
    for d_num=1:in_dim,
        if (mod(d_num,max(1,round(in_dim/50))) == 0)
            fprintf('.');
        end
        Xd = X(:,:);
        if (kill_diags == 1)
            % If diagonals to be killed, preclude a variable from regressing
            % onto itself.
            Xd(:,d_num) = 0;
        end
        if (sparsity < 0.99)
            % Use L1-regularized regression if a sparse fit is desired
            fit = glmnet(Xd, Y(:,d_num), 'gaussian', options);
            mean_basis(d_num,:) = fit.beta(:,end)';
            for j=2:numel(fit.lambda),
                if ((fit.df(j) / in_dim) > sparsity)
                    mean_basis(d_num,:) = fit.beta(:,j-1)';
                    break
                end
            end
        else
            % Do a regular linear regression if a dense fit is desired
            mean_basis(d_num,:) = (pinv(Xd) * Y(:,d_num))';
        end
    end
    fprintf('\n');
    % Find the residuals after accounting for the mean coefficients
    Y_res = Y - (mean_basis * X')';
    % Set residual to ZMUV for each feature
    Y_res = ZMUV(Y_res);
else
    % Skip regressing for the mean
    mean_basis = zeros(in_dim, in_dim);
    Y_res = Y(:,:);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute a set of random projection bases with which to perform regression
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if (rp_count > 0)
    % Use random projections to reduce the size of lwr basis set
    bases_reg = zeros(in_dim,in_dim,rp_count);
    for i=1:rp_count,
        bases_reg(:,:,i) = randn(in_dim,in_dim);
    end
else
    % Use the full identity basis for lwr
    bases_reg = zeros(in_dim,in_dim,in_dim*in_dim);
    for i=1:in_dim,
        for j=1:in_dim,
            bases_reg(i,j,(i-1)*in_dim+j) = 1;
        end
    end
end
rp_count = size(bases_reg,3);
if (kill_diags == 1)
    % Kill diagonal entries of the regression bases if so desired
    for i=1:rp_count,
        for j=1:in_dim,
            bases_reg(j,j,i) = 0;
        end
    end
end

% Perform lwr on the residuals using bases in bases_reg
[ beta ] = lwr_matrix_sparse( X, Y_res, bases_reg, k, 1.0, clip, drop, idx );
% Find principal components of the initial parameter set
[pc,score,latent,tsquare] = princomp(beta);
l_sums = cumsum(latent) / sum(latent);
% X_idx = X(idx,:);
% [ X_hat ] = lwr_predict_matrix( X_idx, bases_reg, beta );
% X_err = X_idx - X_hat;

% Reconstitute the approximate PPC bases from the RP bases and the lwr PCs
ppc_bases = zeros(in_dim,in_dim,rp_count);
for b_num=1:rp_count,
    % Get the loadings for a given principal component
    loadings = pc(:,b_num);
    ppc_basis = zeros(in_dim,in_dim);
    for l_idx=1:numel(loadings),
        % Use a loading-weighted sum to compute an approximate PPC basis
        ppc_basis = ppc_basis + (loadings(l_idx) * squeeze(bases_reg(:,:,l_idx)));
    end
    ppc_bases(:,:,b_num) = ppc_basis;
end

return

end

