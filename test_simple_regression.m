% Setup control variables
obs_count = 7500;
train_count = round(obs_count*(2/3));
train_idx = 1:train_count;
test_idx = train_count+1:obs_count;
sigma_count = 2;
obs_dim = 20;
sigma_spars = 0.66;
min_seg_len = 15;
max_seg_len = 20;
k = 5.0;

% Generate a sequence of observations
[X, sigmas, beta] = make_varcov_seq(obs_count, sigma_count, obs_dim,...
    sigma_spars, min_seg_len, max_seg_len);

% Get the 'class' of each observation
B_class = zeros(obs_count,1);
B_class(beta(:,1) == 1) = 1;
B_class(beta(:,2) == 1) = 2;

% Compute the range of each segment of constant class
B_segments = [];
seg_start = 1;
current_class = B_class(1);
for i=2:obs_count,
    if (current_class ~= B_class(i))
        B_segments = [B_segments; seg_start i-1];
        seg_start = i;
        current_class = B_class(i); 
    end
    if (i == obs_count)
        B_segments = [B_segments; seg_start obs_count];
    end
end
seg_count = size(B_segments,1);

% Compute a basis for each segment of constant class
fprintf('Computing bases per seg:');
A_seg = zeros(obs_dim, obs_dim, obs_count);
for i=1:seg_count,
    if (mod(i,max(1,round(seg_count/50))) == 0)
        fprintf('.');
    end
    seg_range = B_segments(i,1):B_segments(i,2);
    X_seg = X(seg_range,:);
    [ A_hat ] = l1_reg_raw(X_seg, sigma_spars, 0, 0);
    for j=seg_range,
        A_seg(:,:,j) = A_hat(:,:);
    end
end
fprintf('\n');

% Compute a basis for each observation, using lwr
fprintf('Computing bases per obs:');
A_obs = zeros(obs_dim, obs_dim, obs_count);
for i=1:obs_count,
    if (mod(i,max(1,round(obs_count/50))) == 0)
        fprintf('.');
    end
    [ A_hat ] = l1_reg_raw(X, sigma_spars, k, i);
    A_obs(:,:,i) = A_hat(:,:);
end
fprintf('\n');

% Get flattened basis matrices per segment and per observation
A_segf = zeros(obs_count, obs_dim*obs_dim);
A_obsf = zeros(obs_count, obs_dim*obs_dim);
for i=1:obs_count,
    A_segf(i,:) = reshape(squeeze(A_seg(:,:,i)),1,obs_dim*obs_dim);
    A_obsf(i,:) = reshape(squeeze(A_obs(:,:,i)),1,obs_dim*obs_dim);
end

% Compute principal components of the per observation bases, using only the
% training portion of the sequence
[ A_obsf_pc, score, latent, tsquare ] = princomp(A_obsf(train_idx,:));
l_sums = cumsum(latent) / sum(latent);

% Get test observation sets
pc_count = 5;
A_segf_test = A_segf(test_idx,:);
A_obsf_test = A_obsf(test_idx,:);
A_obsf_test_pc = A_obsf_test * A_obsf_pc(:,1:pc_count);

% Compute logistic regression coefficients on the flattened basis matrices, 
% using only the training part of the sequence
[ cv_err_seg beta_lr_seg ] = log_reg_cv( A_segf(train_idx,:),...
    B_class(train_idx), 100, 0.1, 0.1, 1 );
[ cv_err_obs beta_lr_obs ] = log_reg_cv( A_obsf(train_idx,:),...
    B_class(train_idx), 100, 0.1, 0.1, 1 );
[ cv_err_obs_pc beta_lr_obs_pc ] = log_reg_cv( score(train_idx,1:pc_count),...
    B_class(train_idx), 100, 1.0, 0.1, 1 );

% Test logistic regression on the test sets
pred = [A_segf_test ones(size(A_segf_test,1),1)] * beta_lr_seg;
c1_err = sum((pred > 0) & (B_class(test_idx) == 1));
c2_err = sum((pred < 0) & (B_class(test_idx) == 2));
segf_err = (c1_err + c2_err) / numel(test_idx);
fprintf('SEG LR TEST ERROR: %.4f\n', segf_err);
pred = [A_obsf_test ones(size(A_obsf_test,1),1)] * beta_lr_obs;
c1_err = sum((pred > 0) & (B_class(test_idx) == 1));
c2_err = sum((pred < 0) & (B_class(test_idx) == 2));
obsf_err = (c1_err + c2_err) / numel(test_idx);
fprintf('OBS LR TEST ERROR: %.4f\n', obsf_err);
pred = [A_obsf_test_pc ones(size(A_obsf_test_pc,1),1)] * beta_lr_obs_pc;
c1_err = sum((pred > 0) & (B_class(test_idx) == 1));
c2_err = sum((pred < 0) & (B_class(test_idx) == 2));
obsf_pc_err = (c1_err + c2_err) / numel(test_idx);
fprintf('OBSF-PC LR TEST ERROR: %.4f\n', obsf_pc_err);


% EYE BUFFER
