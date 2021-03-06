function [ X, sigmas, beta ] = make_varcov_seq(...
    obs_count, sigma_count, dim, spars, min_inter, max_inter, blur_sigma )
% Make a sequence of obs_count observations, with intervals of length min_inter
% through max_inter of constant covariance. The covariance is selected at random
% from the sigma_count sigmas generated by rand_sparse_cov.

MIN_EIG = 0.2;
gauss_filt = normpdf(-3*blur_sigma:3*blur_sigma, 0, blur_sigma);

% Generate the random sparse covariance matrices
sigmas = zeros(dim, dim, sigma_count);
for i=1:sigma_count,
    sigma = rand_sparse_cov(dim, spars, MIN_EIG, 1);
    sigmas(:,:,i) = sigma(:,:);
end

% Generate the weight sequence
beta = zeros(obs_count, sigma_count);
end_idx = 0;
while (end_idx < obs_count),
    start_idx = end_idx + 1;
    end_idx = start_idx + randi([min_inter, max_inter]);
    if (end_idx > obs_count),
        end_idx = obs_count;
    end
    sigma_idx = randi(sigma_count);
    beta(start_idx:end_idx,sigma_idx) = 1;
end
for i=1:sigma_count,
    beta(:,i) = conv(beta(:,i),gauss_filt,'same');
end
% Generate the observation sequence
X = zeros(obs_count, dim);
for i=1:obs_count,
    b = beta(i,:) ./ sum(beta(i,:));
    beta(i,:) = b(:);
    sigma = zeros(dim,dim);
    for j=1:sigma_count,
        sigma = sigma + (squeeze(sigmas(:,:,j)) .* b(j));
    end
    % Set the observation to zero mean
    mu = zeros(1, dim);
    X(i,:) = mvnrnd(mu, sigma);
end

return

end

