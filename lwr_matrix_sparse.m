function [ beta ] = lwr_matrix_sparse( X, Y, A, k, sparsity, clip, drop, idx )
% Do locally-weighted regression on the observations in X onto the outputs in Y,
% with a Gaussian kernel having bandwidth of k. Assume that the rows in X
% represent a temporal sequence of observations, with a distance of 1 between
% adjacent rows of X. Find a parameter vector for each time point in X. 
%
% Restrict each locally-weighted regression to be described by a linear
% combination of projections onto the basis matrices in A. Use L1-regularized 
% regression for sparsity induction.
%
% Parameters:
%   X: inputs for the lwr (obs_count x in_dim)
%   Y: outputs for the lwr (obs_count x out_dim)
%   A: basis matrices for the lwr (in_dim x in_dim x basis_count)
%   k: Gaussian kernel width (i.e. std) for the lwr
%   sparsity: the maximum allowable rate of non-zeros in lwr coefficients
%   clip: whether or not to include only observations up to the current one
%   drop: whether or not to drop the current observation during lwr
%   idx: the indices into X/Y at which to perform lwr
%
% Output:
%   beta: the learned lwr coefficients for each input/output pair in X/Y
%

MIN_LWR_OBS = 20;
MAX_STEP = round(k * 2.5);
if ~exist('clip','var')
    clip = 0;
end
if ~exist('drop','var')
    drop = 0;
end
if ~exist('idx','var')
    idx = 1:size(X,1);
end

obs_count = numel(idx);
in_dim = size(X,2);
out_dim = size(Y,2);
basis_count = size(A,3);

if (size(Y,1) ~= size(X,1))
    error('lwr_matrix_sparse: mismatched input/output obs counts!\n');
end
if (max(idx) > size(X,1))
    error('lwr_matrix_sparse: desired lwr index out of range!\n');
end
if (in_dim ~= out_dim)
    error('lwr_matrix_sparse: mismatched input/output obs dimensions!\n');
end
if (size(A,1) ~= in_dim || size(A,2) ~= in_dim)
    error('lwr_matrix_sparse: mismatched basis/input dimensions!\n');
end

bases = {};
for i=1:basis_count,
    bases{i} = squeeze(A(:,:,i));
end

% Setup the options structure for glmnet
options = glmnetSet();
options.dfmax = max(10,ceil(sparsity * in_dim));

% Compute either the sparse or dense lwr for each time point
beta = zeros(obs_count, basis_count);
fprintf('Computing sparse lwr:');
for obs_num=1:obs_count,
    obs_idx = idx(obs_num);
    if (mod(obs_num, round(obs_count/50)) == 0),
        fprintf('.');
    end
    % Set the range of observations to be used in this lwr
    start_step = max(obs_idx-MAX_STEP,1);
    end_step = min(obs_idx+MAX_STEP,size(X,1));
    if (clip == 1)
        end_step = max(obs_idx,MIN_LWR_OBS);
    end
    steps = start_step:end_step;
    % Filter out the current observation if so desired
    if (drop == 1)
        steps = steps(steps ~= obs_idx);
    end
    % Compute normalized kernel weights for these observations
    weights = normpdf(steps-obs_idx, 0, k);
    weights = weights ./ sum(weights);
    weights = diag(sqrt(weights));
    % Project the observation for this step onto the bases
    X_s = weights * X(steps,:);
    X_w = zeros(numel(steps)*in_dim,basis_count);
    for b=1:basis_count,
        Xb = bases{b} * X_s';
        X_w(:,b) = reshape(Xb,numel(steps)*in_dim,1);
    end
    Y_s = (weights * Y(steps,:))';
    Y_w = reshape(Y_s,numel(steps)*in_dim,1);
    % Perform the kernel-weighted, l1-regularized regression on observations
    if (sparsity < 0.99)
        % Do an L1-regularized regression when a sparse fit is desired
        fit = glmnet(X_w, Y_w, 'gaussian', options);
        beta(obs_num,:) = fit.beta(:,end)';
        for j=2:numel(fit.lambda),
            if ((fit.df(j) / basis_count) > sparsity)
                beta(obs_num,:) = fit.beta(:,j-1)';
                break
            end
        end
    else
        % Do a simple linear regression when a dense solution is desired
        b = X_w \ Y_w;
        beta(obs_num,:) = b';
    end
end
fprintf('\n');

return

end

