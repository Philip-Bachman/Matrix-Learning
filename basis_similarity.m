function [ sim_matrix ] = basis_similarity( A, A_hat )
% Check a basic measure of the similarity between the matrices in A and A_hat,
% where similarity is measured relative to a basis' self-similarity
%
% Parameters:
%   A: set of bases to consider true covariance matrices
%   A_hat: set of bases to consider approximate precision matrices
%
% Output:
%   sim_matrix: n x m similarity matrix for the n matrices in A and m matrices
%               in A_hat
%

sim_matrix = zeros(size(A,3), size(A_hat,3));
for i=1:size(A,3),
    A_mat = pinv(squeeze(A(:,:,i)));
    A_mat = offdiag_zmuv(A_mat);
    A_selfsim = sum(sum(A_mat.*A_mat));
    % Check matrix-matrix correlation for each approximate precision matrix in
    % A_hat, and normalize by self-similarity of A_mat
    for j=1:size(A_hat,3),
        Ah_mat = offdiag_zmuv(squeeze(A_hat(:,:,j)));
        Ah_sim = sum(sum(A_mat.*Ah_mat));
        sim_matrix(i,j) = Ah_sim / A_selfsim;
    end
end

return

end

function [ zmuv_mat ] = offdiag_zmuv(mat)
% Set the off-diagonal entries of a matrix to ZMUV and set diagonals to zero.
mat = mat - diag(diag(mat));
% Do zmuving
nz_vals = find(mat);
mat(nz_vals) = mat(nz_vals) - mean(mat(nz_vals));
zmuv_mat = mat ./ std2(mat);

return

end


    
    
