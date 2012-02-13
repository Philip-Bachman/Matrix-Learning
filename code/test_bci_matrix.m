% Run tests on some preloaded BCI data.
% in_name should be one of:
%   {'my-c3iiia-k3b.mat','my-c3iiia-k6b.mat','my-c3iiia-l1b.mat'}
% out_name should be one of:
%   {'my-results-k3b.mat','my-results-k6b.mat','my-results-l1b.mat'}
% 
clear;

%% =============================================================================
% BEGINNING OF DATA LOADING AND PREPROCESSING
%===============================================================================

in_name = 'my-c3iiia-l1b.mat';
out_name = 'my-results-l1b.mat';

load(in_name);
warning off all;

trial_count = all_trials.trial_count;
trials = all_trials.trials;

% Count the amount of data to be loaded and init arrays to hold it
obs_count = 0;
obs_dim = 0;
for t_num=1:trial_count,
    tr = trials{t_num};
    if (tr.true_label == 1 || tr.true_label == 2)
        obs_count = obs_count + size(tr.filt_data,1);
        obs_dim = size(tr.filt_data,2);
    end
end
X_12 = zeros(obs_count,obs_dim);
Y_12 = zeros(obs_count,1);
T_12 = zeros(obs_count,1);

tr_labels = [];
tr_range_idx = [];

c_end = 0;
% Load data from all trials into a pair of arrays
for t_num=1:trial_count,
    % Get the current trial
    tr = trials{t_num};
    tr_obs_count = size(tr.filt_data,1);
    % Only process trials in class one and two
    if (tr.true_label == 1 || tr.true_label == 2)
        c_offset = c_end + 1;
        c_end = c_offset + tr_obs_count - 1;
        tr_range_idx = [tr_range_idx; c_offset c_end];
        % Get the observation and class data for this trial
        X_12(c_offset:c_end,:) = ZMUV(tr.filt_data(:,:));
        Y_12(c_offset:c_end) = tr.true_label;
        T_12(c_offset:c_end) = numel(tr_labels) + 1;
        tr_labels = [tr_labels; tr.true_label];
    end
end

% Set observations (i.e. EEG readings) to ZMUV
X_12 = ZMUV(X_12);

% Find the ten worst trials, with respect to the (kurtosis of the) log of the
% probability of their observations, under a Gaussian model for the data.
X_12_mahal = mahal(X_12,X_12);
tr_mahal_k = zeros(size(tr_range_idx,1),1);
for i=1:size(tr_mahal_k,1),
    tr_mahal_k(i) = kurtosis(X_12_mahal(tr_range_idx(i,1):tr_range_idx(i,2)));
end
[mahalk_vals mahalk_idx] = sort(tr_mahal_k,'descend');
keep_tr_idx = mahalk_idx(11:end);
keep_obs_idx = [];
for i=1:numel(keep_tr_idx),
    tr_old_start = tr_range_idx(keep_tr_idx(i),1);
    tr_old_end = tr_range_idx(keep_tr_idx(i),2);
    tr_new_start = numel(keep_obs_idx) + 1;
    keep_obs_idx = [keep_obs_idx tr_old_start:tr_old_end];
    tr_new_end = numel(keep_obs_idx);
    tr_range_idx(keep_tr_idx(i), 1) = tr_new_start;
    tr_range_idx(keep_tr_idx(i), 2) = tr_new_end;
end

% Filter the data to remove the ten worst trials by the computed metric
X_12 = X_12(keep_obs_idx,:);
Y_12 = Y_12(keep_obs_idx,1);
T_12 = T_12(keep_obs_idx,1);
tr_labels = tr_labels(keep_tr_idx);
tr_range_idx = tr_range_idx(keep_tr_idx,:);
trial_count = size(tr_range_idx,1);

% Clear unneeded vars
clearvars -except ...
    'X_12' 'Y_12' 'T_12' 'tr_labels' 'tr_range_idx' 'in_name' 'out_name';

% Compute a whitening transform for X_12.
% The whitening transform is V*(D^(-1/2)), where columns of V are the
% eigenvectors of cov(X_12) and D is a diagonal matrix comprising the
% corresponding eigenvalues of cov(X_12). After projection onto the eigenvectors
% and rescaling (i.e. the whitening transform), we project the back into the
% original representation space, to allow capture of sparse network structure
% among the EEG electrodes.
[V D] = eigs(cov(X_12),size(X_12,2));
% Project onto eigenvectors and rescale by sqrt(eigenvalues)
X_12w = X_12 * (V*D^(-1/2));
% Rotate data back into original representation (i.e. electrode space)
X_12w = X_12w * V';

%% =============================================================================
% END OF LOADING/PREPROCESSING, BEGINNING OF BASIS LEARNING AND CODING
%===============================================================================

% Learn the PPC initializing bases via random projections
fprintf('Learning PPC bases:\n');
train_idx = randsample(size(X_12w,1), 10000);
rp_count = 200;
sparse1 = 0.5;
k = 15.0;
% Learn a set of (random-projection-approximate) PPC bases
[ ppc_bases mean_basis l_sums ] = learn_matrix_bases_ppc(...
    X_12w, X_12w, k, sparse1, 0, 0, 1, rp_count, train_idx, 1 );

% Get the residuals after accounting for the mean basis, and set them to ZMUV
X_12r = X_12w - (mean_basis * X_12w')';
fprintf('MEAN BASIS EXPLAINS %.4f OF VARIANCE\n',...
    (1 - sum(var(X_12r))/sum(var(X_12w))));
X_12r = ZMUV(X_12r);

% Take the first few bases as a starting point for stochastic updates
basis_count = 40;
ppcb = ppc_bases(:,:,1:basis_count);
ppcb_pre = ppcb(:,:,:);

% Kill basis diagonal entries, and normalize to zero mean, unit variance
for i=1:size(ppcb,3),
    basis = squeeze(ppcb(:,:,i));
    for j=1:size(basis,1),
        basis(j,j) = 0;
    end
    basis = basis ./ std(basis(:));
    ppcb(:,:,i) = basis;
end

%% =============================================================================
% Do stochastic gradient descent updates of the approximate PPC bases
% On first pass, don't use an L1 penalty and set the wobble/noise sort of high
%===============================================================================
step_size = 200.0;
sparse2 = 0.8;
l1_pen = 0.0;
kill_diags = 1;
k = 15.0;
noise_lvl = 0.33;
fprintf('FIRST PASS UPDATES:\n');
for i=1:100,
    train_idx = randsample(size(X_12r,1),6000);
    beta_ppcb = lwr_matrix_sparse(...
        X_12r, X_12r, ppcb, k, sparse2, 0, 0, train_idx );
    X_train = X_12r(train_idx,:);
    [ ppcb_t post_err pre_err step_taken ] = update_bases(...
        ppcb, beta_ppcb, X_train, X_train, step_size, l1_pen, kill_diags, noise_lvl );
    fprintf('pre_err: %.4f post_err: %.4f, step: %.4f, kurtosis: %.4f\n',...
        pre_err, post_err, step_taken, kurtosis(ppcb_t(:)));
    ppcb = ppcb_t(:,:,:);
    noise_lvl = noise_lvl * 0.98;
end

% Second pass, apply an L1 penalty and set the wobble/noise a bit lower
l1_pen = 0.0001;
noise_lvl = 0.2;
fprintf('SECOND PASS UPDATES:\n');
for i=1:100,
    train_idx = randsample(size(X_12r,1),6000);
    beta_ppcb = lwr_matrix_sparse(...
        X_12r, X_12r, ppcb, k, sparse2, 0, 0, train_idx );
    X_train = X_12r(train_idx,:);
    [ ppcb_t post_err pre_err step_taken ] = update_bases(...
        ppcb, beta_ppcb, X_train, X_train, step_size, l1_pen, kill_diags, noise_lvl );
    fprintf('pre_err: %.4f post_err: %.4f, step: %.4f, kurtosis: %.4f\n',...
        pre_err, post_err, step_taken, kurtosis(ppcb_t(:)));
    ppcb = ppcb_t(:,:,:);
    noise_lvl = noise_lvl * 0.98;
end


%% =============================================================================
% Code the set of class 1/2 observations using the bases we just learned
%===============================================================================
train_idx = [0];
block_size = 15000;
final_idx = size(X_12r,1);
B_12 = zeros(final_idx,size(ppcb,3));
while 1,
    block_start = max(train_idx) + 1;
    block_end = min(final_idx, max(train_idx)+block_size);
    train_idx = block_start:block_end;
    fprintf('CODING OBS %d TO %d OF %d:\n', block_start, block_end, final_idx);
    B_part = lwr_matrix_sparse(...
        X_12r, X_12r, ppcb, k, sparse2, 0, 0, train_idx );
    B_12(block_start:block_end,:) = B_part(:,:);
    if (block_end == final_idx)
        % Break when the final observation has been coded
        break
    end
end

B_12F = ZMUV(B_12);

save(out_name);

%% =============================================================================
% END OF CODING, BEGINNING OF CLASSIFICATION TESTING
%===============================================================================

trial_count = size(tr_range_idx,1);
test_frac = 0.2; % Fraction of trials to put in test set
good_times = 500:1250; % Consider data from cue time to 2.5 sec after cue
win_starts = [125,250,375,500]; % Windows at half-second post-cue intervals
win_step = 250; % Windows extend over a one second period
num_filt = 5;
 
%==============================================================================
% Test the graphy features, using means aggregated over multiple windows
%==============================================================================
fprintf('============================================================\n');
fprintf('+ TESTING GRAPHY FEATURES                                  +\n');
fprintf('============================================================\n');
features_graph = [];
for tr_num=1:trial_count,
    tr_start = tr_range_idx(tr_num,1);
    tr_gt = good_times + (tr_start-1);
    tr_fv = [];
    for ws=win_starts,
        wb_means = mean(B_12F(tr_gt(ws:ws+win_step),:));
        wb_vars = var(B_12F(tr_gt(ws:ws+win_step),:));
        tr_fv = [tr_fv wb_means];
    end
    features_graph = [features_graph; tr_fv];
end

% Standardize the feature vectors we just computed
features_graph = ZMUV(features_graph);

% Test the features via cross-validated l1-regularized logistic regression
spars = [0.02, 0.04 0.06 0.08 0.10 0.15 0.20 0.25 0.30 0.35];
fprintf('Testing sparse graph features:\n');
for s=spars,
    [cv_err beta_mean] = log_reg_cv(features_graph, tr_labels, 250, s, test_frac, 1);
    fprintf('spars=%.4f: mean=%.4f, min=%.4f, max=%.4f, std=%.4f\n',s, mean(cv_err), min(cv_err), max(cv_err), std(cv_err));
end

%% =============================================================================
% Compute a set of RCSP filters using the trial data using a range of
% regularization weights, checking the utility of the features generated using
% each regularization weight.
%===============================================================================
fprintf('============================================================\n');
fprintf('+ COMPUTING/TESTING RCSP FILTERS                           +\n');
fprintf('============================================================\n');
alphas = [0.0 0.05 0.1 0.15 0.20];
spars = [0.2 0.4 0.6];
min_err = 1.1;
min_features = [];
rounds = 20;
rcsp_cv_errs = zeros(numel(alphas),numel(spars),rounds);
for a_num=1:numel(alphas),
    alpha = alphas(a_num);
    for s_num=1:numel(spars),
        s = spars(s_num);
        fprintf('=================================================\n');
        fprintf('Testing RCSP @ alpha=%.2f, sparse=%.2f:\n', alpha, s);
        for r=1:rounds,
            test_idx = randsample(trial_count,round(trial_count*test_frac));
            X1_train = [];
            X2_train = [];
            for t=1:trial_count,
                if ~ismember(t,test_idx)
                    tr_start = tr_range_idx(t,1);
                    tr_gt = good_times + (tr_start-1);
                    if (tr_labels(t) == 1)
                        X1_train = [X1_train; X_12w(tr_gt,:)];
                    else
                        X2_train = [X2_train; X_12w(tr_gt,:)];
                    end
                end
            end
            % Compute a set of rcsp filters using the training set
            rcsp_filters = compute_rcsp_filters(X1_train, X2_train, num_filt, alpha);
            X_rcsp = X_12w * rcsp_filters;
            X_test = [];
            Y_test = [];
            X_train = [];
            Y_train = [];
            for t=1:trial_count,        
                tr_start = tr_range_idx(t,1);
                tr_gt = good_times + (tr_start-1);
                tr_fv = [];
                for ws=win_starts,
                    x_vars = var(X_rcsp(tr_gt(ws:ws+win_step),:));
                    tr_fv = [tr_fv log(x_vars)];
                end
                if ~ismember(t,test_idx)
                    X_train = [X_train; tr_fv];
                    Y_train = [Y_train; tr_labels(t)];
                else
                    X_test = [X_test; tr_fv];
                    Y_test = [Y_test; tr_labels(t)];
                end
            end
            X_train = ZMUV(X_train);
            X_test = ZMUV(X_test);
            % Test the features via l1-regularized logistic regression
            [cv_err beta_rcsp] = log_reg_cv(X_train, Y_train, 50, s, 0.05, 1);
            Y_pred_rcsp = [X_test ones(size(X_test,1),1)] * beta_rcsp;
            c1_misses = sum((Y_pred_rcsp > 0) & (Y_test == 1));
            c2_misses = sum((Y_pred_rcsp < 0) & (Y_test == 2));
            err = (c1_misses + c2_misses) / numel(Y_test);
            fprintf('err=%.4f\n', err);
            rcsp_cv_errs(a_num,s_num,r) = err;
        end
    end
end

%% =============================================================================
% Test RCSP filters and threshold-based graphy classifier
%===============================================================================
stream = RandStream.getDefaultStream();
c = clock();
reset(stream,round(1000*c(6)));


B_12F = ZMUV(B_12);
rounds = 50;
graphy_err = zeros(rounds,1);
gm_err = zeros(rounds,1);
gv_err = zeros(rounds,1);
thresh_err = zeros(rounds,1);
rcsp_err = zeros(rounds,1);
joint_lr_err = zeros(rounds,1);
joint_lda_err = zeros(rounds,1);
alpha = 0.1;
blur_filt = normpdf(-40:40,0,10.0);
C1_cumsums = [];
C2_cumsums = [];
beta_mean_thresh = zeros(41,1);
beta_mean_gm = zeros(161,1);
for r=1:rounds,
    fprintf('TEST ROUND %d\n', r);
    % Randomly select test examples from each class to make a 0.8/0.2 split
    test_size = round(trial_count / 10);
    test_idx = [randsample(find(tr_labels == 1),test_size) randsample(find(tr_labels == 2),test_size)];
    X1_train = [];
    X2_train = [];
    B_train = [];
    B_train_tr_idx = [];
    B_train_tr_labels = [];
    Y_train = [];
    B_test = [];
    B_test_tr_idx = [];
    B_test_tr_labels = [];
    Y_test = [];
    %
    % Extract training and testing data for graphy filters and RCSP
    %
    for t=1:trial_count,
        % Training data is taken only from 'good_times', while testing data will
        % be taken comprise complete trials (though classification will only
        % consider test data from 'good_times').
        tr_start = tr_range_idx(t,1);
        tr_end = tr_range_idx(t,2);
        tr_at = tr_start:tr_end;
        tr_gt = good_times + tr_start;
        if ~ismember(t,test_idx)
            if (tr_labels(t) == 1)
                X1_train = [X1_train; X_12w(tr_gt,:)];
            else
                X2_train = [X2_train; X_12w(tr_gt,:)];
            end
            tr_start = size(B_train,1) + 1;
            tr_end = tr_start + numel(tr_gt) - 1;
            B_train = [B_train; B_12F(tr_gt,:)];
            Y_train = [Y_train; round(ones(numel(tr_gt),1).*tr_labels(t))];
            B_train_tr_idx = [B_train_tr_idx; tr_start:tr_end];
            B_train_tr_labels = [B_train_tr_labels; tr_labels(t)];
        else
            tr_start = size(B_test,1) + 1;
            tr_end = tr_start + numel(tr_at) - 1;
            B_test = [B_test; B_12F(tr_at,:)];
            Y_test = [Y_test; round(ones(numel(tr_at),1).*tr_labels(t))];
            B_test_tr_idx = [B_test_tr_idx; tr_start:tr_end];
            B_test_tr_labels = [B_test_tr_labels; tr_labels(t)];
        end
    end
    %
    % Compute a beta using non-smooothed graphy features
    %
    fprintf('Fitting beta_sums...\n');
    %[cv_err beta_sums] = log_reg_cv(...
    %    B_train, Y_train, 20, 0.5, 0.5, 1);
    beta_sums = glmfit([B_train ones(size(B_train,1),1)],...
        round(Y_train-1),'binomial','constant','off');
    beta_mean_thresh = beta_mean_thresh + beta_sums;
    Y_pred_sums = [B_test ones(size(B_test,1),1)] * beta_sums;
    Y_pred_sums = conv(Y_pred_sums,blur_filt,'same');
    c1_err = sum((Y_pred_sums > 0) & (Y_test == 1));
    c2_err = sum((Y_pred_sums < 0) & (Y_test == 2));
    graphy_err(r) = (c1_err + c2_err) / numel(Y_test);
    %
    % Get cumsums to illustrate cumulative evidence per-trial
    %
    B_test_tr_cs = [];
    for t=1:numel(B_test_tr_labels),
        tr_vals = Y_pred_sums(B_test_tr_idx(t,:));
        tr_vals = exp(tr_vals) ./ (exp(tr_vals) + 1);
        tr_vals = tr_vals - 0.5;
        cs = cumsum(tr_vals);
        if (B_test_tr_labels(t) == 1)
            C1_cumsums = [C1_cumsums; reshape(cs,1,numel(cs))];
        else
            C2_cumsums = [C2_cumsums; reshape(cs,1,numel(cs))];
        end
    end
    % Check classification via threshold on cumulative evidence
    [ best_thresh best_err ] = get_cumev_thresh(...
        beta_sums, B_train, B_train_tr_idx, B_train_tr_labels );
    Y_pred_thresh = zeros(numel(B_test_tr_labels),1);
    fprintf('Thresh train err: %.4f\n', best_err);
    sum_range = good_times(125:end);
    for t=1:numel(B_test_tr_labels),
        tr_sum = mean(Y_pred_sums(B_test_tr_idx(t,sum_range)));
        Y_pred_thresh(t) = tr_sum;
    end
    c1_err = sum((Y_pred_thresh > best_thresh) & (B_test_tr_labels == 1));
    c2_err = sum((Y_pred_thresh < best_thresh) & (B_test_tr_labels == 2));
    thresh_err(r) = (c1_err + c2_err) / numel(B_test_tr_labels);
    %
    % Compute a set of rcsp filters using the training set
    %
    rcsp_filters = compute_rcsp_filters(X1_train, X2_train, num_filt, alpha);
    X_rcsp = X_12w * rcsp_filters;
    X_test_rcsp = [];
    X_test_gv = [];
    X_test_gm = [];
    Y_test_rcsp = [];
    X_train_rcsp = [];
    X_train_gv = [];
    X_train_gm = [];
    Y_train_rcsp = [];
    for t=1:trial_count,        
        tr_start = tr_range_idx(t,1);
        tr_gt = good_times + (tr_start-1);
        tr_fv = [];
        tr_fvv = [];
        tr_fvm = [];
        for ws=win_starts,
            x_vars = var(X_rcsp(tr_gt(ws:ws+win_step),:));
            tr_fv = [tr_fv log(x_vars)];
            b_mean = mean(B_12F(tr_gt(ws:ws+win_step),:));
            tr_fvm = [tr_fvm b_mean];
        end
        if ~ismember(t,test_idx)
            X_train_rcsp = [X_train_rcsp; tr_fv];
            Y_train_rcsp = [Y_train_rcsp; tr_labels(t)];
            X_train_gm = [X_train_gm; tr_fvm];
        else
            X_test_rcsp = [X_test_rcsp; tr_fv];
            Y_test_rcsp = [Y_test_rcsp; tr_labels(t)];
            X_test_gm = [X_test_gm; tr_fvm];
        end
    end
    % Jointly standardize graphy and rcsp features
    temp = ZMUV([X_train_rcsp; X_test_rcsp]);
    X_train_rcsp = temp(1:numel(Y_train_rcsp),:);
    X_test_rcsp = temp(numel(Y_train_rcsp)+1:end,:);
    temp = ZMUV([X_train_gm; X_test_gm]);
    X_train_gm = temp(1:numel(Y_train_rcsp),:);
    X_test_gm = temp(numel(Y_train_rcsp)+1:end,:);
    %
    % Test the RCSP features via l1-regularized logistic regression
    %
    [cv_err beta_rcsp] = log_reg_cv(...
        X_train_rcsp, Y_train_rcsp, 50, 0.4, 0.1, 1);
    fprintf('Rcsp train err: %.4f\n', mean(cv_err));
    Y_pred_rcsp = [X_test_rcsp ones(size(X_test_rcsp,1),1)] * beta_rcsp;
    c1_err = sum((Y_pred_rcsp > 0) & (Y_test_rcsp == 1));
    c2_err = sum((Y_pred_rcsp < 0) & (Y_test_rcsp == 2));
    rcsp_err(r) = (c1_err + c2_err) / numel(Y_test_rcsp);
    %
    % Test the graphy features via l1-regularized logistic regression
    %
    [cv_err beta_gm] = log_reg_cv(...
        X_train_gm, Y_train_rcsp, 50, 0.3, 0.1, 1);
    fprintf('Graphy (means) train err: %.4f\n', mean(cv_err));
    beta_mean_gm = beta_mean_gm + beta_gm;
    Y_pred_gm = [X_test_gm ones(size(X_test_gm,1),1)] * beta_gm;
    c1_err = sum((Y_pred_gm > 0) & (Y_test_rcsp == 1));
    c2_err = sum((Y_pred_gm < 0) & (Y_test_rcsp == 2));
    gm_err(r) = (c1_err + c2_err) / numel(Y_test_rcsp);
    %
    % Test combination of threshold and RCSP filter classifier
    %
    X_train_rcsp = [X_train_rcsp ones(size(X_train_rcsp,1),1)] * beta_rcsp;
    X_train_gm = [X_train_gm ones(size(X_train_gm,1),1)] * beta_gm;
    X_train_sums = [B_train ones(size(B_train,1),1)] * beta_sums;
    X_train_thresh = zeros(numel(B_train_tr_labels),1);
    sum_range = 125:min(875,size(B_train_tr_idx,2));
    for t=1:numel(B_train_tr_labels),
        tr_sum = mean(X_train_sums(B_train_tr_idx(t,sum_range)));
        X_train_thresh(t) = tr_sum;
    end
    % Jointly standardize the training and test set
    X_train_thresh = [X_train_thresh; Y_pred_thresh] - best_thresh;
    X_train_rcsp = [X_train_rcsp; Y_pred_rcsp];
    X_train_gm = [X_train_gm; Y_pred_gm];
    X_train_thresh = ZMUV(X_train_thresh);
    X_train_rcsp = ZMUV(X_train_rcsp);
    X_train_gm = ZMUV(X_train_gm);
    % Recover training/test sets from the jointly standardized arrays
    X_test_thresh = X_train_thresh(numel(Y_train_rcsp)+1:end,1);
    X_train_thresh = X_train_thresh(1:numel(Y_train_rcsp),1);
    X_test_rcsp = X_train_rcsp(numel(Y_train_rcsp)+1:end,1);
    X_train_rcsp = X_train_rcsp(1:numel(Y_train_rcsp),1);
    X_test_gm = X_train_gm(numel(Y_train_rcsp)+1:end,1);
    X_train_gm = X_train_gm(1:numel(Y_train_rcsp),1);
    % Create joint traning/testing sets for an rcsp/graphy classifier
    X_train_joint = [X_train_gm X_train_rcsp];
    Y_train_joint = Y_train_rcsp(:,1);
    X_test_joint = [X_test_gm X_test_rcsp];
    Y_test_joint = Y_test_rcsp(:,1);
    % Train the joint rcsp/graphy classifier
    [cv_err beta_joint] = log_reg_cv(...
        X_train_joint, Y_train_joint, 50, 1.1, 0.1, 1);
    % Test the joint rcsp/graphy classifier
    Y_pred_joint = [X_test_joint ones(size(X_test_joint,1),1)] * beta_joint;
    c1_err = sum((Y_pred_joint > 0) & (Y_test_joint == 1));
    c2_err = sum((Y_pred_joint < 0) & (Y_test_joint == 2));
    joint_lr_err(r) = (c1_err + c2_err) / numel(Y_test_joint);
    Y_pred_joint = classify(X_test_joint, X_train_joint, Y_train_joint, 'quadratic');
    joint_lda_err(r) = sum((Y_pred_joint ~= Y_test_joint)) / numel(Y_test_joint);
    fprintf('ERRORS: GM=%.4f, RCSP=%.4f, THRESH=%.4f, J-LR=%.4f, J-LDA=%.4f\n',...
        gm_err(r), rcsp_err(r), thresh_err(r), joint_lr_err(r), joint_lda_err(r));
    fprintf('    JOINT BETA: [%.4f %.4f %.4f]\n',...
        beta_joint(1), beta_joint(2), beta_joint(3));
end

beta_mean_thresh = beta_mean_thresh ./ rounds;
beta_mean_gm = beta_mean_gm ./ rounds;


%% Plot cumsums
figure();
hold on;
h = plot(mean(C1_cumsums)+std(C1_cumsums),'b:');
set(get(get(h,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
h = plot(mean(C1_cumsums)-std(C1_cumsums),'b:');
set(get(get(h,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
h = plot(mean(C2_cumsums)+std(C2_cumsums),'r:');
set(get(get(h,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
h = plot(mean(C2_cumsums)-std(C2_cumsums),'r:');
set(get(get(h,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
plot(mean(C1_cumsums),'b-');
plot(mean(C2_cumsums),'r-');




% EYE BUFFER

