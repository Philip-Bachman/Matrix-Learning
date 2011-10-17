% Set some basic parameters
dim = 20;
obs_count = 20000;
sigma_count = 2;
sigma_spars = 0.66;
min_inter = 15;
max_inter = 25;
train_size = round(obs_count * 0.8);
learn_spars = 0.8;
k = 5.0;

[X, sigmas, beta] = make_varcov_seq(obs_count, sigma_count, dim, sigma_spars,...
                    min_inter, max_inter);

Xtr = X(1:train_size,:);
Xte = X(train_size+1:end,:);
Btr = beta(1:train_size,:);
Bte = beta(train_size+1:end,:);


[ppc_bases mean_basis l_sums] = learn_matrix_bases_ppc(Xtr, Xtr, k,...
                                learn_spars, 0, 0, 1, 150, 1:10000, 1);

Xtr_res = Xtr - (mean_basis * Xtr')';
Xte_res = Xte - (mean_basis * Xte')';
fprintf('MEAN BASIS EXPLAINS %.4f OF VARIANCE\n',...
    (1 - sum(var(Xtr_res))/sum(var(Xtr))));
Xtr_res = ZMUV(Xtr_res);
Xte_res = ZMUV(Xte_res);

% Initialize the bases with which to begin updates
ppcb = ppc_bases(:,:,1:6);
% ppcb = randn(dim,dim,6);
% Kill diagonal entries
for i=1:size(ppcb,3),
    for j=1:size(ppcb,1),
        ppcb(j,j,i) = 0;
    end
end
% Normalize bases to unit variance
for i=1:size(ppcb,3),
    ppcb(:,:,i) = ppcb(:,:,i) ./ std(reshape(ppcb(:,:,i),numel(ppcb(:,:,i)),1));
end

%% =============================================================================
% Do stochastic gradient descent updates of the approximate PPC bases
% On first pass, don't use an L1 penalty and set the wobble/noise sort of high
%===============================================================================
step_size = 150.0;
sparse2 = 0.8;
l1_pen = 0.0;
kill_diags = 1;
k = 3.0;
noise_lvl = 0.25;
fprintf('FIRST PASS UPDATES:\n');
for i=1:50,
    train_idx = randsample(size(Xtr_res,1),6000);
    beta_ppcb = lwr_matrix_sparse(Xtr_res, Xtr_res, ppcb, k, sparse2,...
        0, 0, train_idx);
    [ ppcb_t post_err pre_err ] = update_bases(ppcb, beta_ppcb,...
        Xtr_res(train_idx,:), Xtr_res(train_idx,:), step_size, l1_pen,...
        kill_diags, noise_lvl);
    fprintf('round: %d, pre_err: %.4f post_err: %.4f, basis change: %.4f, kurtosis: %.4f\n',...
        i, pre_err, post_err, var(ppcb(:)-ppcb_t(:))/var(ppcb(:)), kurtosis(ppcb_t(:)));
    ppcb = ppcb_t(:,:,:);
    noise_lvl = noise_lvl * 0.98;
end

%% =============================================================================
% Do second pass updates, with more L1 penalty and less wobble/noise
%===============================================================================
l1_pen = 0.0001;
noise_lvl = 0.25;
fprintf('SECOND PASS UPDATES:\n');
for i=1:150,
    train_idx = randsample(size(Xtr_res,1),6000);
    beta_ppcb = lwr_matrix_sparse(Xtr_res, Xtr_res, ppcb, k, sparse2,...
        0, 0, train_idx);
    [ ppcb_t post_err pre_err ] = update_bases(ppcb, beta_ppcb,...
        Xtr_res(train_idx,:), Xtr_res(train_idx,:), step_size, l1_pen,...
        kill_diags, noise_lvl);
    fprintf('round: %d, pre_err: %.4f post_err: %.4f, basis change: %.4f, kurtosis: %.4f\n',...
        i, pre_err, post_err, var(ppcb(:)-ppcb_t(:))/var(ppcb(:)), kurtosis(ppcb_t(:)));
    ppcb = ppcb_t(:,:,:);
    noise_lvl = noise_lvl * 0.98;
end

% Encode the training set using the learned bases
[ Btr_hat ] = lwr_matrix_sparse(Xtr_res, Xtr_res, ppcb, k, 0.75,...
    0, 0);

% Set up classification targets for the training and test sets
Btr_class = zeros(train_size,1);
for i=1:train_size,
    if (Btr(i,1) == 1),
        Btr_class(i) = 1;
    else
        Btr_class(i) = 2;
    end
end
Bte_class = zeros(size(Bte,1),1);
for i=1:numel(Bte_class),
    if (Bte(i,1) == 1),
        Bte_class(i) = 1;
    else
        Bte_class(i) = 2;
    end
end

% Learn a logistic regression classifier with the training set codes
[ cv_err beta_lr ] = log_reg_cv( Btr_hat, Btr_class, 200, 1.0, 0.5, 1 );

% Check the error of the learned lr coefficients on the training set
Btr_pred = [Btr_hat ones(size(Btr_hat,1),1)] * beta_lr;
c1_err = sum((Btr_pred > 0) & (Btr_class == 1));
c2_err = sum((Btr_pred < 0) & (Btr_class == 2));
train_err = (c1_err + c2_err) / numel(Btr_class);
fprintf('LR TRAINING ERROR: %.4f\n', train_err);

% Encode the test set using the learned bases
[ Bte_hat ] = lwr_matrix_sparse(Xte_res, Xte_res, ppcb, k, 0.75,...
    0, 0);

% Check performance of the learned lr coefficients on the test set
Bte_pred = [Bte_hat ones(size(Bte_hat,1),1)] * beta_lr;
c1_err = sum((Bte_pred > 0) & (Bte_class == 1));
c2_err = sum((Bte_pred < 0) & (Bte_class == 2));
test_err = (c1_err + c2_err) / numel(Bte_class);
fprintf('LR TESTING ERROR: %.4f\n', test_err);

                            