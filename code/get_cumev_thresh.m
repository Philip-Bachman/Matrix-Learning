function [ best_thresh best_err ] = get_cumev_thresh( beta, B, B_tr_idx, B_tr_labels )
% Get a classification threshold for the data in B_train.

% Sum over evidence from 0.5s after cue to (up to) 3.5s after cue
sum_range = 125:min(875,size(B_tr_idx,2));
% Blur evidence to de-jitter
blur_filt = normpdf(-40:40,0,10.0);
% Get number of trials in the training set
trial_count = numel(B_tr_labels);

Y_pred = [B ones(size(B,1),1)] * beta;
Y_pred = conv(Y_pred,blur_filt,'same');

Y_sums = zeros(trial_count,1);
for t=1:trial_count,
    t_sum = mean(Y_pred(B_tr_idx(t,sum_range)));
    Y_sums(t) = t_sum;
end

threshs = linspace(min(Y_sums),max(Y_sums),1000);
thresh_count = numel(threshs);
thresh_errs = zeros(thresh_count,1);
rounds = 30;
for r=1:rounds,
    % Do a bootstrapped test of possible thresholds to reduce overfitting
    idx = randsample(1:trial_count,trial_count,true);
    for t=1:thresh_count,
        thresh = threshs(t);
        c1_err = sum((Y_sums(idx) > thresh) & (B_tr_labels(idx) == 1));
        c2_err = sum((Y_sums(idx) < thresh) & (B_tr_labels(idx) == 2));
        thresh_err = (c1_err + c2_err) / trial_count;
        thresh_errs(t) = thresh_errs(t) + thresh_err;
    end
end
thresh_errs = thresh_errs ./ rounds;     

[errs_sorted errs_idx] = sort(thresh_errs,'ascend');

best_thresh = threshs(errs_idx(1));
best_err = thresh_errs(errs_idx(1));

return

end

