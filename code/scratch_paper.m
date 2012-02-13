obs_dims = 10:3:40;
dim_count = numel(obs_dims);
sigma_count = 2;
obs_count = 1000;

round_count = 15;

sim_scores = zeros(round_count, dim_count);
for round_num=1:round_count,
    fprintf('ROUND %d:', round_num);
    for dim_num=1:dim_count,
        obs_dim = obs_dims(dim_num);
        fprintf(' %d', obs_dim);
        [ X, sigmas, beta ] = make_varcov_seq(obs_count, sigma_count, obs_dim,...
            sigma_spars, 10, 15, 3.0 );
        sigmas_pinv = zeros(size(sigmas));
        for j=1:sigma_count,
            sigmas_pinv(:,:,j) = pinv(squeeze(sigmas(:,:,j)));
        end
        [ sim_matrix ] = basis_similarity( sigmas, sigmas_pinv );
        % Record the geometric mean of the self-similarity scores
        sim_scores(round_num, dim_num) = geomean(diag(sim_matrix));
    end
    fprintf('\n');
end