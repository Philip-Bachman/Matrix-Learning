function [ ws ] = rand_walk_weights( obs_count, obs_dim, sigma, step_size )
% Make a set of weights using smoothed independent univariate random walks with
% smoothing performed by convolution with a Gaussian having std sigma

% Make the sequence using a random-walk metropolis hastings sampling of a
% gaussian distribution with unit variance. Random-walk step proposals will be
% drawn from a Normal distribution with modest variance, to induce a high
% autocorrelation, which is desired in this case.
if ~exist('step_size','var')
    step_size = 0.20;
end

ws = zeros(obs_count, obs_dim);
w_old = randn(1, obs_dim);
for i=1:obs_count,
    w_new = w_old + (step_size * randn(1, obs_dim));
    new_prob = prod(normpdf(w_new, 0.0, 1.0));
    old_prob = prod(normpdf(w_old, 0.0, 1.0));
    a = rand();
    if (a < min(new_prob/old_prob, 1))
        w_old = w_new;
    end
    ws(i,:) = w_old;
end

% Set up the smoothing kernel
gauss_kernel = zeros(1, round(4*sigma) + 1);
for i=1:numel(gauss_kernel),
    gauss_kernel(i) = normpdf(i - numel(gauss_kernel)/2 + 0.5, 0.0, sigma);
end

% Do the convolutions to smooth the weight walks
for i=1:obs_dim,
    w_walk = ws(:,i);
    w_walk = conv(w_walk, gauss_kernel, 'same');
    ws(:,i) = w_walk;
end

return

end

