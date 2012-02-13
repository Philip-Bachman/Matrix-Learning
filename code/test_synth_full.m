%
% Test different values for the smoothing parameter k, for the lwrs and PCA
% prior to basis updating.
%
clear;
stream = RandStream.getDefaultStream();
c = clock();
reset(stream,round(1000*c(6)));

% Do full testing with synthetic data
round_count = 5;
k_vals = 3.0:0.5:6.0;
obs_dims = [15]; %[10 15 20 25 30 35 40];
k_count = numel(k_vals);
dim_count = numel(obs_dims);

% Setup control variables
obs_count = 9000;
train_count = round(obs_count*(2/3));
train_idx = 1:train_count;
lr_tr_idx = randsample(train_count, round(train_count/10));
lr_tr_idx = train_idx(lr_tr_idx);
test_idx = train_count+1:obs_count;
sigma_count = 4;
blur_sigma = 3.0;
sigma_spars = 0.33;
sigma_reg = 0.5;
min_seg_len = 10;
max_seg_len = 20;

sim_results_pc = zeros(round_count, dim_count);
sim_results_adapt = zeros(round_count, dim_count);
sim_results_true = zeros(round_count, dim_count);
class_errs_raw = zeros(round_count, dim_count);
class_errs_pc = zeros(round_count, dim_count);
class_errs_adapt = zeros(round_count, dim_count);

%
% Run tests, set k and obs_dim to default values
%
obs_dim = 16;
for round_num=1:round_count,
    fprintf('============================================================\n');
    fprintf('STARTING ROUND %d\n', round_num);
    fprintf('============================================================\n');
    for dim_num=1:dim_count,
        k = 5.0;
        obs_dim = obs_dims(dim_num);
        fprintf('============================================================\n');
        fprintf('TESTING DIMENSION %d\n', obs_dim);
        fprintf('============================================================\n');
        %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % GENERATE SEQUENCE AND COMPUTE BASIC RAW/PC
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        % Generate a sequence of observations
        [X, sigmas, beta] = make_varcov_seq(obs_count, sigma_count, obs_dim,...
            sigma_spars, min_seg_len, max_seg_len, blur_sigma);
        
        % Compute the precision bases from the generated sigmas
        sigmas_prec = zeros(size(sigmas));
        for i=1:sigma_count,
            sigmas_prec(:,:,i) = pinv(squeeze(sigmas(:,:,i)));
        end

        % Get the 'class' of each observation
        B_class = zeros(obs_count,1);
        B_class(beta(:,1)+beta(:,2) >= beta(:,3)+beta(:,4)) = 1;
        B_class(B_class == 0) = 2;

        % Compute a basis A_t for each observation, using lwr
        fprintf('Computing bases per obs:');
        Ats = zeros(obs_dim, obs_dim, obs_count);
        for i=1:obs_count,
            if (mod(i,max(1,round(obs_count/50))) == 0)
                fprintf('.');
            end
            [ At ] = l1_reg_raw(X, sigma_reg, k, i);
            Ats(:,:,i) = At(:,:);
        end
        fprintf('\n');

        % Get flattened basis matrices per segment and per observation
        Ats_f = zeros(obs_count, obs_dim*obs_dim);
        for i=1:obs_count,
            Ats_f(i,:) = reshape(squeeze(Ats(:,:,i)),1,obs_dim*obs_dim);
        end

        % Compute principal components of the per observation bases, using only
        % the training portion of the sequence
        [ pc, score, latent, tsquare ] = princomp(Ats_f(train_idx,:));
        mean_basis = reshape(mean(Ats_f(train_idx,:)),obs_dim,obs_dim);
        Xr = X - (mean_basis * X')';

        % Get test observation sets
        pc_count = 6;
        Ats_f_pc = bsxfun(@minus, Ats_f, reshape(mean_basis,1,obs_dim*obs_dim))...
            * pc(:,1:pc_count);
        Ats_f_test = Ats_f(test_idx,:);
        Ats_f_test_pc = Ats_f_pc(test_idx,:);
        
        bases_pc = zeros(obs_dim, obs_dim, pc_count);
        for i=1:pc_count,
            basis = reshape(pc(:,i),obs_dim,obs_dim);
            for d=1:obs_dim,
                basis(d,d) = 0;
            end
            basis = basis ./ std2(basis);
            bases_pc(:,:,i) = basis(:,:);
        end

        %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Update the principal component bases via block coordinate descent
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        k = 4.0;
        bases_adapt = zeros(size(bases_pc));
        for i=1:pc_count,
            basis = squeeze(bases_pc(:,:,i));
            basis = basis + randn(obs_dim,obs_dim).*(0.3*std2(basis));
            for d=1:obs_dim,
                basis(d,d) = 0;
            end
            basis = (basis + basis') ./ 2;
            basis = basis ./ std2(basis);
            bases_adapt(:,:,i) = basis(:,:);
        end

        % Control variables for block coordinate descent
        step_size = 25;
        do_cv = 0;
        kill_diags = 1;
        noise_lvl = 0.2;
        
        % Do basis updates
        fprintf('UPDATING PC BASES:\n');
        lwr_spars = 0.66;
        % Use no l1 penalty on basis entries
        l1_pen = 0.0;
        bases_adapt = learn_sggm_bases(X(train_idx,:), pc_count, k, lwr_spars,...
            l1_pen, step_size, 30, bases_adapt);
        % Use a small l1 penalty on basis entries
        l1_pen = 0.0005 * (10 / obs_dim);
        bases_adapt = learn_sggm_bases(X(train_idx,:), pc_count, k, lwr_spars,...
            l1_pen, step_size, 30, bases_adapt);
        % Encode the full data set using the updated bases
        beta_adapt = lwr_matrix_sparse(X, X, bases_adapt, k, lwr_spars, 0, 0);
        beta_adapt_test = beta_adapt(test_idx,:);
        
        %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % TEST BASIS QUALITY
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        % Check similarity of PC-projected bases to true bases
        [basis_sims_pc] = basis_similarity(sigmas, bases_pc);
        best_sims_pc = max(abs(basis_sims_pc'));
        % Check similarity of learned bases to true bases
        [basis_sims_adapt] = basis_similarity(sigmas, bases_adapt);
        best_sims_adapt = max(abs(basis_sims_adapt'));
        % Check similarity of true bases to true bases, for reference
        [basis_sims_true] = basis_similarity(sigmas, sigmas_prec);
        best_sims_true = max(abs(basis_sims_true'));
        
        % Compute logistic regression coefficients on the flat bases
        [ cv_err_raw beta_lr_raw ] = log_reg_cv( Ats_f(lr_tr_idx,:),...
            B_class(lr_tr_idx), 100, 0.1, 0.1, 1 );
        % Learn a logistic regression classifier using PC projected bases
        [ cv_err_pc beta_lr_pc ] = log_reg_cv( Ats_f_pc(lr_tr_idx,1:pc_count),...
            B_class(lr_tr_idx), 100, 1.0, 0.1, 1 );
        % Learn a logistic regression classifier using learned bases
        [ cv_err_adapt beta_lr_adapt ] = log_reg_cv( beta_adapt(lr_tr_idx,:),...
            B_class(lr_tr_idx), 100, 1.0, 0.1, 1 );

        % Test logistic regression on flat bases
        pred = [Ats_f_test ones(size(Ats_f_test,1),1)] * beta_lr_raw;
        c1_err = sum((pred > 0) & (B_class(test_idx) == 1));
        c2_err = sum((pred < 0) & (B_class(test_idx) == 2));
        err_raw = (c1_err + c2_err) / numel(test_idx);
        % Test logistic regression on PC-projected bases
        pred = [Ats_f_test_pc ones(size(Ats_f_test_pc,1),1)] * beta_lr_pc;
        c1_err = sum((pred > 0) & (B_class(test_idx) == 1));
        c2_err = sum((pred < 0) & (B_class(test_idx) == 2));
        err_pc = (c1_err + c2_err) / numel(test_idx);
        % Test logistic regression on learned bases
        pred = [beta_adapt_test ones(size(beta_adapt_test,1),1)] * beta_lr_adapt;
        c1_err = sum((pred > 0) & (B_class(test_idx) == 1));
        c2_err = sum((pred < 0) & (B_class(test_idx) == 2));
        err_adapt = (c1_err + c2_err) / numel(test_idx);
        
        % Display regression scores
        fprintf('RAW TEST ERROR: %.4f\n', err_raw);
        fprintf('PC TEST ERROR: %.4f\n', err_pc);
        fprintf('ADAPT TEST ERROR: %.4f\n', err_adapt);
        
        % Record results for this test round
        sim_results_pc(round_num,dim_num) = geomean(best_sims_pc);
        sim_results_adapt(round_num,dim_num) = geomean(best_sims_adapt);
        sim_results_true(round_num,dim_num) = geomean(best_sims_true);
        class_errs_raw(round_num,dim_num) = err_raw;
        class_errs_pc(round_num,dim_num) = err_pc;
        class_errs_adapt(round_num,dim_num) = err_adapt;
    end
    % Save results in case of need for partial results
    save('synth_test_xxx.mat');
end

% EYE BUFFER


