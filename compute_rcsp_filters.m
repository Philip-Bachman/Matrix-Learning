function [ rcsp_filters ] = compute_rcsp_filters(X1, X2, num_filt, alpha)
% Compute 2 x num_filts RCSP filters, using identity-matrix Tikhonov (i.e. L2)
% regularization on the filter coefficients. The regularization weight used will
% be alpha and num_filt filters for each class will be generated.
%
% Parameters:
%   X1: input observations for class 1 (x1_count x obs_dim)
%   X2: input observations for class 2 (x2_count x obs_dim)
%   num_filt: the number of filters to compute for each class
%   alpha: the regularization weight
%
% Output:
%   rcsp_filters: the learned filters (obs_dim x 2*num_filt)
%

obs_dim = size(X1,2);
if (size(X2,2) ~= obs_dim)
    error('compute_rcsp_filters: obs_dim mistmatch in X1/X2!\n');
end

I = eye(obs_dim);

% Compute the covariance matrix for class 1 data
C1 = cov(X1);
% Compute the covariance matrix for class 2 data
C2 = cov(X2);

rcsp_filters = zeros(obs_dim, num_filt*2);
% Compute num_filt filters for class 1 versus class 2
M1 = pinv(C2 + alpha * I) * C1;
[V1,D1] = eig(M1);
rcsp_filters(:,1:num_filt) = V1(:,1:num_filt);
% Compute num_filt filters for class 2 versus class 1
M2 = pinv(C1 + alpha * I) * C2;
[V2,D2] = eig(M2);
rcsp_filters(:,num_filt+1:num_filt+num_filt) = V2(:,1:num_filt);

return

end

