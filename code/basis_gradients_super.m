function [ dA_un dA_su ] = basis_gradients_super( A, b, w, x_in, x_out, y,...
    l1_bases, l2_reg, l_mix )
% Get the supervised element-wise basis gradients for:
%   L(...) =   l_mix * ||x_out - sum_i(b(i)*A(i)*x_in)||^2
%            + (1 - l_mix) * l(b'*w, y)
%            + l1_bases * sum_i ||A(i)||_1
%
% Parameters:
%   A: a collection of bases onto which inputs were projected prior to lwr.
%   b: lwr coefficients for each of the bases in A (basis_count x 1)
%   w: classification coefficients (basis_count x 1)
%   x_in: the input observation (always a vector) (in_dim x 1)
%   x_out: the output observation (always a vector) (out_dim x 1)
%   y: target class/output, to be passed to a (differentiable) loss function
%   l1_bases: l1 sparsifying penalty to place on basis entries (scalar)
%   l2_reg: l2 penalty used in the elastic-net regression which produced b
%   l_mix: mixing ratio for supervised/unsupervised gradients
%
% Outputs:
%   dA_un: gradients of the objective with respect to each basis entry
%

if (nargin < 9)
    error('basis_gradients_super(): All parameters required\n');
end
in_dim = size(x_in,1);
out_dim = size(x_out,1);
basis_count = size(b,1);
b_act = find(abs(b) > 1e-5);

if (l_mix > 0)
    % First, compute the "unsupervised", reconstruction-based gradients
    D = zeros(out_dim,basis_count); % pseudo-dictionary for this x_in
    for i=b_act',
        D(:,i) = squeeze(A(:,:,i)) * x_in;
    end
    dD = -2 * (x_out - D*b) * b';
    dA_un = backprop_dict_grads(dD, x_in, b_act);
else
    dA_un = zeros(in_dim,in_dim,basis_count);
end

if (l_mix < 1)
    % Second, compute the supervised part of gradients
    [L dLdF] = loss_bindev(b'*w, y, 1, 0);
    B = zeros(basis_count,1);
    I = eye(length(b_act));
    B(b_act) = (D(:,b_act)'*D(:,b_act) + l2_reg*I) \ (dLdF * w(b_act));
    dD = (-D * B * b') + ((x_out - (D * b)) * B');
    dA_su = backprop_dict_grads(dD, x_in, b_act);
else
    dA_su = zeros(in_dim,in_dim,basis_count);
end

return

end

function [ dA ] = backprop_dict_grads(dD, x_in, b_act)
% Backpropagate gradients that have been computed for each element of each
% pseudo-dictionary element in D, such that D(:,i) = A(:,:,i) * x_in.
basis_count = size(dD,2);
in_dim = size(dD,1);
dA = zeros(in_dim,in_dim,basis_count);
for i=b_act',
        dA(:,:,i) = dD(:,i) * x_in';
end
return
end



