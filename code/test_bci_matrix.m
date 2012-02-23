% Run tests on some preloaded BCI data.
% in_name should be one of:
%   {'my-c3iiia-k3b.mat','my-c3iiia-k6b.mat','my-c3iiia-l1b.mat'}
% out_name should be one of:
%   {'my-results-k3b.mat','my-results-k6b.mat','my-results-l1b.mat'}

clear;

%===============================================================================
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
Y_12 = round((Y_12 - 1.5) .* 2);

%===============================================================================
% END OF LOADING/PREPROCESSING, BEGINNING OF BASIS LEARNING AND CODING
%===============================================================================

trial_count = size(tr_range_idx,1);
good_times = 500:1250; % Consider data from cue time to 2.5 sec after cue
train_times = [];
for i=1:trial_count,
    train_times = [train_times; good_times' + tr_range_idx(i,1)];
end

% Learn the PPC initializing bases via random projections
fprintf('Learning PPC bases:\n');
train_idx = randsample(train_times, 10000);
rp_count = 100;
sparse1 = 0.5;
k = 10.0;
% Learn a set of (random-projection-approximate) PPC bases
[ ppc_bases mean_basis l_sums ] = learn_matrix_bases_ppc(...
    X_12w, X_12w, k, sparse1, 0, 0, 1, rp_count, train_idx, 1 );

% Get the residuals after accounting for the mean basis, and set them to ZMUV
X_12r = X_12w - (mean_basis * X_12w')';
fprintf('MEAN BASIS EXPLAINS %.4f OF VARIANCE\n',...
    (1 - sum(var(X_12r))/sum(var(X_12w))));
X_12r = ZMUV(X_12r);

% Take the first few bases as a starting point for stochastic updates
basis_count = 20;
ppcb = ppc_bases(:,:,1:basis_count);

% Kill basis diagonal entries, and normalize to zero mean, unit variance
for i=1:size(ppcb,3),
    basis = squeeze(ppcb(:,:,i));
    for j=1:size(basis,1),
        basis(j,j) = 0;
    end
    basis = basis ./ std(basis(:));
    ppcb(:,:,i) = basis;
end

%===============================================================================
% Do stochastic gradient descent updates of the approximate PPC bases
% On first pass, don't use an L1 penalty and set the wobble/noise sort of high
%===============================================================================

% Setup basis learning options
%   opts: structure determining the following options:
%     basis_count: number of basis matrices to learn
%     k: kernel width for locally-weighted regressions
%     spars: desired sparsity for locally-weighted regressions
%     l1_bases: l1 penalty to use for basis entries
%     l_mix: mixing ratio (1 -> unsuper only ... 0 -> super only)
%     step: initial step size for gradient descent
%     round_count: number of update rounds to perform
%     Ai: optional starting basis matrices
%     wi: optional starting classifier coefficients
%     idx: optional indices into X/Y to use in updates
lrn_opts = struct();
lrn_opts.basis_count = basis_count;
lrn_opts.k = 4.0;
lrn_opts.spars = 0.5;
lrn_opts.l1_bases = 0.0;
lrn_opts.l_mix = 1.0;
lrn_opts.noise_lvl = 0.25;
lrn_opts.step = 20;
lrn_opts.round_count = 5;
lrn_opts.Ai = ppcb;
fprintf('FIRST PASS UPDATES:\n');
for i=1:50,
    train_idx = randsample(train_times,1000);
    lrn_opts.idx = train_idx;
    % Update learning opts, for updated bases and coefficients
    lrn_opts.Ai = ppcb;
    ppcb = learn_bases_super(X_12r, Y_12, lrn_opts);
end

% Second pass, apply an L1 penalty and set the wobble/noise a bit lower
lrn_opts.l1_bases = 1e-4;
lrn_opts.l_mix = 0.8;
lrn_opts.step = 10;
noise_lvl = 0.2;
fprintf('SECOND PASS UPDATES:\n');
for i=1:10,
    train_idx = randsample(train_times,1000);
    lrn_opts.idx = train_idx;
    % Encode the full data set using the updated bases
    b_ppcb = lwr_matrix_sparse(...
        X_12r, X_12r, ppcb, lrn_opts.k, lrn_opts.spars, 0, 0, train_idx);
    % Update logistic regression coefficients
    w_ppcb = wl2_logreg(b_ppcb, Y_12(train_idx), 1e-3, 0, zeros(basis_count,1), 250);
    pred = b_ppcb * w_ppcb;
    err = sum(pred.*Y_12(train_idx) < 0) / length(train_idx);
    fprintf('CLASS ERR: %.4f\n', err);
    % Update learning opts, for updated bases and coefficients
    lrn_opts.wi = w_ppcb;
    lrn_opts.Ai = ppcb;
    ppcb = learn_bases_super(X_12r, Y_12, lrn_opts);
end
save(out_name); % Save learned bases, in case of early code death
% 
% %===============================================================================
% % Code the set of class 1/2 observations using the bases we just learned
% %===============================================================================
% train_idx = [0];
% block_size = 15000;
% final_idx = size(X_12r,1);
% B_12 = zeros(final_idx,size(ppcb,3));
% while 1,
%     block_start = max(train_idx) + 1;
%     block_end = min(final_idx, max(train_idx)+block_size);
%     train_idx = block_start:block_end;
%     fprintf('CODING OBS %d TO %d OF %d:\n', block_start, block_end, final_idx);
%     B_part = lwr_matrix_sparse(...
%         X_12r, X_12r, ppcb, k, sparse1, 0, 0, train_idx );
%     B_12(block_start:block_end,:) = B_part(:,:);
%     if (block_end == final_idx)
%         % Break when the final observation has been coded
%         break
%     end
% end
% 
% B_12F = ZMUV(B_12);
% 

%===============================================================================
% END OF CODING, BEGINNING OF CLASSIFICATION TESTING
%===============================================================================

trial_count = size(tr_range_idx,1);
test_frac = 0.2; % Fraction of trials to put in test set
win_starts = [125,250,375,500]; % Windows at half-second post-cue intervals
win_step = 250; % Windows extend over a one second period
num_filt = 10;
 
% %==============================================================================
% % Test the graphy features, using means aggregated over multiple windows
% %==============================================================================
% fprintf('============================================================\n');
% fprintf('+ TESTING GRAPHY FEATURES                                  +\n');
% fprintf('============================================================\n');
% features_graph = [];
% for tr_num=1:trial_count,
%     tr_start = tr_range_idx(tr_num,1);
%     tr_gt = good_times + (tr_start-1);
%     tr_fv = [];
%     for ws=win_starts,
%         wb_means = mean(B_12F(tr_gt(ws:ws+win_step),:));
%         wb_vars = var(B_12F(tr_gt(ws:ws+win_step),:));
%         tr_fv = [tr_fv wb_means];
%     end
%     features_graph = [features_graph; tr_fv];
% end
% 
% % Standardize the feature vectors we just computed
% features_graph = ZMUV(features_graph);
% 
% % Test the features via cross-validated l1-regularized logistic regression
% spars = [0.10 0.15 0.20 0.25 0.30 0.35];
% fprintf('Testing sparse graph features:\n');
% for s=spars,
%     [cv_err beta_mean] = log_reg_cv(features_graph, tr_labels, 250, s, test_frac, 1);
%     fprintf('spars=%.4f: mean=%.4f, min=%.4f, max=%.4f, std=%.4f\n',s, mean(cv_err), min(cv_err), max(cv_err), std(cv_err));
% end
% 
% %% =============================================================================
% % Compute a set of RCSP filters using the trial data using a range of
% % regularization weights, checking the utility of the features generated using
% % each regularization weight.
% %===============================================================================
% fprintf('============================================================\n');
% fprintf('+ COMPUTING/TESTING RCSP FILTERS                           +\n');
% fprintf('============================================================\n');
% alphas = [0.0 0.05 0.1];
% spars = [0.4 0.6 0.8];
% min_err = 1.1;
% min_features = [];
% rounds = 20;
% rcsp_cv_errs = zeros(numel(alphas),numel(spars),rounds);
% for a_num=1:numel(alphas),
%     alpha = alphas(a_num);
%     for s_num=1:numel(spars),
%         s = spars(s_num);
%         fprintf('=================================================\n');
%         fprintf('Testing RCSP @ alpha=%.2f, sparse=%.2f:\n', alpha, s);
%         for r=1:rounds,
%             test_idx = randsample(trial_count,round(trial_count*test_frac));
%             X1_train = [];
%             X2_train = [];
%             for t=1:trial_count,
%                 if ~ismember(t,test_idx)
%                     tr_start = tr_range_idx(t,1);
%                     tr_gt = good_times + (tr_start-1);
%                     if (tr_labels(t) == 1)
%                         X1_train = [X1_train; X_12w(tr_gt,:)];
%                     else
%                         X2_train = [X2_train; X_12w(tr_gt,:)];
%                     end
%                 end
%             end
%             % Compute a set of rcsp filters using the training set
%             rcsp_filters = compute_rcsp_filters(X1_train, X2_train, num_filt, alpha);
%             X_rcsp = X_12w * rcsp_filters;
%             X_test = [];
%             Y_test = [];
%             X_train = [];
%             Y_train = [];
%             for t=1:trial_count,        
%                 tr_start = tr_range_idx(t,1);
%                 tr_gt = good_times + (tr_start-1);
%                 tr_fv = [];
%                 for ws=win_starts,
%                     x_vars = var(X_rcsp(tr_gt(ws:ws+win_step),:));
%                     tr_fv = [tr_fv log(x_vars)];
%                 end
%                 if ~ismember(t,test_idx)
%                     X_train = [X_train; tr_fv];
%                     Y_train = [Y_train; tr_labels(t)];
%                 else
%                     X_test = [X_test; tr_fv];
%                     Y_test = [Y_test; tr_labels(t)];
%                 end
%             end
%             X_train = ZMUV(X_train);
%             X_test = ZMUV(X_test);
%             % Test the features via l1-regularized logistic regression
%             [cv_err beta_rcsp] = log_reg_cv(X_train, Y_train, 50, s, 0.1, 1);
%             Y_pred_rcsp = [X_test ones(size(X_test,1),1)] * beta_rcsp;
%             c1_misses = sum((Y_pred_rcsp > 0) & (Y_test == 1));
%             c2_misses = sum((Y_pred_rcsp < 0) & (Y_test == 2));
%             err = (c1_misses + c2_misses) / numel(Y_test);
%             fprintf('err=%.4f\n', err);
%             rcsp_cv_errs(a_num,s_num,r) = err;
%         end
%     end
% end

%===============================================================================
% Test RCSP filters and threshold-based graphy classifier
%===============================================================================
stream = RandStream.getDefaultStream();
c = clock();
reset(stream,round(1000*c(6)));

rounds = 15;
thresh_err = zeros(rounds,1);
rcsp_err = zeros(rounds,1);
joint_err = zeros(rounds,1);
alpha = 0.05;
blur_filt = normpdf(-10:10,0,3.0);
C1_cumsums = [];
C2_cumsums = [];
beta_mean_thresh = zeros(basis_count, 1);
for r=1:rounds,
    fprintf('TEST ROUND %d\n', r);
    % Randomly select test examples from each class to make a 0.8/0.2 split
    test_size = round(trial_count / 10);
    test_idx = [randsample(find(tr_labels == 1),test_size) randsample(find(tr_labels == 2),test_size)];
    X_train = [];
    train_tr_idx = [];
    train_tr_labels = [];
    Y_train = [];
    X_test = [];
    test_tr_idx = [];
    test_tr_labels = [];
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
            tr_start = size(X_train,1) + 1;
            tr_end = tr_start + numel(tr_gt) - 1;
            X_train = [X_train; X_12w(tr_gt,:)];
            Y_train = [Y_train; Y_12(tr_gt)];
            train_tr_idx = [train_tr_idx; tr_start:tr_end];
            train_tr_labels = [train_tr_labels; tr_labels(t)];
        else
            tr_start = size(X_test,1) + 1;
            tr_end = tr_start + numel(tr_at) - 1;
            X_test = [X_test; X_12w(tr_at,:)];
            Y_test = [Y_test; Y_12(tr_at)];
            test_tr_idx = [test_tr_idx; tr_start:tr_end];
            test_tr_labels = [test_tr_labels; Y_12(tr_at(1))];
        end
    end
    % Adapt unsupervised bases using supervisory feedback
    lrn_opts.l1_bases = 1e-3;
    lrn_opts.l_mix = 0.75;
    lrn_opts.step = 10;
    noise_lvl = 0.2;
    fprintf('SUPERVISED UPDATES:\n');
    ppcb_tr = ppcb;
    for i=1:5,
        train_idx = randsample(size(X_train,1),1000);
        lrn_opts.idx = train_idx;
        % Encode the full data set using the updated bases
        b_ppcb = lwr_matrix_sparse(...
            X_train, X_train, ppcb, lrn_opts.k, lrn_opts.spars, 0, 0, train_idx);
        % Update logistic regression coefficients
        w_ppcb = wl2_logreg(b_ppcb, Y_train(train_idx), 1e-3, 0, zeros(basis_count,1), 250);
        pred = b_ppcb * w_ppcb;
        err = sum(pred.*Y_train(train_idx) < 0) / length(train_idx);
        fprintf('CLASS ERR: %.4f\n', err);
        % Update learning opts, for updated bases and coefficients
        lrn_opts.wi = w_ppcb;
        lrn_opts.Ai = ppcb_tr;
        ppcb_tr = learn_bases_super(X_train, Y_train, lrn_opts);
    end
    % Recode a subsampled set of the training observations with the new
    % bases, to facilitate classifier learning.
    train_idx = randsample(size(X_train,1),5000);
    B_train = lwr_matrix_sparse(X_train, X_train, ppcb_tr,...
        lrn_opts.k, lrn_opts.spars, 0, 0, train_idx);
    % Recode the full set of test observations
    B_test = lwr_matrix_sparse(X_test, X_test, ppcb_tr,...
        lrn_opts.k, lrn_opts.spars, 0, 0);
    %
    % Compute a classifier based on network structure features
    %
    fprintf('Fitting beta_thresh...\n');
    beta_thresh = wl2_logreg(B_train, Y_train(train_idx), 1e-3);
    beta_mean_thresh = beta_mean_thresh + (beta_thresh ./ rounds);
    Y_pred_thr = B_test * beta_thresh;
    fprintf('Single time-point thresh err: %.4f\n', ...
        (sum(Y_pred_thr .* Y_test < 0) / numel(Y_test)));
    %
    % Get cumsums to illustrate cumulative evidence per-trial
    %
    B_test_tr_cs = [];
    for t=1:numel(test_tr_labels),
        tr_vals = Y_pred_thr(test_tr_idx(t,:));
        tr_vals = exp(tr_vals) ./ (exp(tr_vals) + 1);
        tr_vals = tr_vals - 0.5;
        cs = cumsum(tr_vals);
        if (test_tr_labels(t) < 0)
            C1_cumsums = [C1_cumsums; reshape(cs,1,numel(cs))];
        else
            C2_cumsums = [C2_cumsums; reshape(cs,1,numel(cs))];
        end
    end
    % Check classification via threshold on cumulative evidence
    Y_pred_thresh = zeros(numel(test_tr_labels),1);
    sum_range = good_times(125:end);
    for t=1:numel(test_tr_labels),
        tr_sum = mean(Y_pred_thr(test_tr_idx(t,sum_range)));
        Y_pred_thresh(t) = tr_sum;
    end
    thresh_err(r) = sum(Y_pred_thresh .* test_tr_labels < 0) / ...
        numel(test_tr_labels);
    %
    % Compute a set of rcsp filters using the training set and then compute
    % a "single time-point" classifier based on these features.
    %
    rcsp_filters = compute_rcsp_filters(X_train(Y_train==-1,:),...
        X_train(Y_train==1,:), num_filt, alpha);
    R_train = (X_train(train_idx,:) * rcsp_filters).^2;
    beta_rcsp = wl2_logreg(R_train, Y_train(train_idx), 1e-4);
    R_test = (X_test * rcsp_filters).^2;
    Y_pred_rcs = R_test * beta_rcsp;
    fprintf('Single time-point rcsp err: %.4f\n', ...
        (sum(Y_pred_rcs .* Y_test < 0) / numel(Y_test)));
    % Check classification via threshold on cumulative evidence
    Y_pred_rcsp = zeros(numel(test_tr_labels),1);
    sum_range = good_times(125:end);
    for t=1:numel(test_tr_labels),
        tr_sum = mean(Y_pred_rcs(test_tr_idx(t,sum_range)));
        Y_pred_rcsp(t) = tr_sum;
    end
    rcsp_err(r) = sum(Y_pred_rcsp .* test_tr_labels < 0) / ...
        numel(test_tr_labels);
    %
    % Test combination of threshold and RCSP filter classifier
    %
    J_train = [R_train B_train];
    J_test = [R_test B_test];
    beta_joint = wl2_logreg(J_train, Y_train(train_idx), 1e-4);
    Y_pred_j = J_test * beta_joint;
    fprintf('Single time-point joint err: %.4f\n', ...
        (sum(Y_pred_j .* Y_test < 0) / numel(Y_test)));
    % Check classification via threshold on cumulative evidence
    Y_pred_joint = zeros(numel(test_tr_labels),1);
    sum_range = good_times(125:end);
    for t=1:numel(test_tr_labels),
        tr_sum = mean(Y_pred_j(test_tr_idx(t,sum_range)));
        Y_pred_joint(t) = tr_sum;
    end
    joint_err(r) = sum(Y_pred_joint .* test_tr_labels < 0) / ...
        numel(test_tr_labels);
    fprintf('ERRORS: RCSP=%.4f, THRESH=%.4f, JOINT=%.4f\n',...
         rcsp_err(r), thresh_err(r), joint_err(r));
    save(out_name);
end

% % Plot cumsums
% figure();
% hold on;
% h = plot(mean(C1_cumsums)+std(C1_cumsums),'b:');
% set(get(get(h,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
% h = plot(mean(C1_cumsums)-std(C1_cumsums),'b:');
% set(get(get(h,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
% h = plot(mean(C2_cumsums)+std(C2_cumsums),'r:');
% set(get(get(h,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
% h = plot(mean(C2_cumsums)-std(C2_cumsums),'r:');
% set(get(get(h,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
% plot(mean(C1_cumsums),'b-');
% plot(mean(C2_cumsums),'r-');

% EYE BUFFER

