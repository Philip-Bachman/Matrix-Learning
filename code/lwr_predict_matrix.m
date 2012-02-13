function [ Y_hat ] = lwr_predict_matrix( X, A, B, A_mean )
% Get the outputs predicted from the inputs X, bases A, and basis weights B.
%
% Parameters:
%   X: input observations (obs_count x in_dim)
%   A: basis matrices (in_dim x in_dim x basis_count)
%   B: basis weights (obs_count x basis_count)
%   A_mean: optional mean matrix
%
% Output:
%   Y_hat: the outputs predicted using the input parameters
%

obs_count = size(X,1);
in_dim = size(X,2);
basis_count = size(A,3);

if (size(A,1) ~= in_dim || size(A,2) ~= in_dim)
    error('lwr_predict_matrix: mismatched basis/input dimensions\n');
end
if (size(B,2) ~= basis_count)
    error('lwr_predict_matrix: mismatched basis counts in A/B\n');
end
if (size(B,1) ~= obs_count)
    error('lwr_predict_matrix: mismatched obs counts in X/B\n');
end

if exist('A_mean','var')
    A(:,:,basis_count+1) = A_mean(:,:);
    B = [B ones(obs_count,1)];
end

Y_hat = zeros(obs_count, in_dim);
X_bmult = zeros(obs_count, in_dim, basis_count);
for i=1:basis_count,
    X_bmult(:,:,i) = (squeeze(A(:,:,i)) * X')';
end
for i=1:obs_count,
    Xb = squeeze(X_bmult(i,:,:));
    Y_hat(i,:) = (Xb * B(i,:)')';
end

return

end

