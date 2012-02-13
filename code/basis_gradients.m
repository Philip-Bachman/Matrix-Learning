function [ A_grads ] = basis_gradients( A, b, x, y, l1_pen )
% Get the element-wise gradients of bases for:
%   ||y - sum_i(b(i)*A(i)*x)||^2 + l1_pen * sum_i ||A(i)||_1
%
% Parameters:
%   A: a collection of bases onto which inputs were projected prior to lwr.
%   b: lwr coefficients for each of the bases in A (basis_count x 1)
%   x: the input observation (always a vector) (in_dim x 1)
%   y: the output observation (always a vector) (out_dim x 1)
%   l1_pen: l1 sparsifying penalty to place on basis entries (scalar, 1 x 1)
%
% Outputs:
%   A_grads: partial derivatives of the objective with respect to each basis
%            entry
%

if ~exist('l1_pen','var')
    l1_pen = 0.0;
end

out_dim = size(y,1);
basis_count = size(b,1);

% Compute basis gradients when each basis is a matrix
if (out_dim ~= size(x,1))
    error('basis_gradients: Dimensions wrong with vector-vector basis gradients!\n');
end
y_hat = zeros(out_dim,basis_count);
basis = zeros(out_dim,out_dim);
for i=1:basis_count,
    basis(:,:) = A(:,:,i);
    y_hat(:,i) = basis * x;
end

y_hat_sum = y_hat * b;
y_dif = y_hat_sum - y;

A_grads = zeros(size(A));
part_grads = y_dif * x';
part_grads = part_grads .* 2.0;

for i=1:basis_count,
    if (abs(b(i)) > 0.00001)
        A_grads(:,:,i) = part_grads .* b(i);
    end
end

% Add an L1 regularization term, to sparsify the learned basis matrices.
A_grads = A_grads + (sign(A) .* l1_pen);

return

end



