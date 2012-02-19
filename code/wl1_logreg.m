function [ w ] = wl1_logreg( X, Y, lam, bias, wi, max_iter )
% Compute a weighted, pseudo-l1-regularized logistic regression using the
% observations in X and weighted classifications in Y.
%
% Parameters:
%   X: the input observations (obs_count x obs_dim)
%   Y: the weighted classifications (obs_count x 1)
%   lam: the regularization weight for pseudo-l1 penalty
%   bias: whether to include a bias term in the regression
%   wi: optional initialization vector for w
%   max_iter: maximum number of minFunc iterations
%
% Outputs:
%   beta: the weights of the regression (obs_dim(+1) x 1)
%

obs_count = size(X,1);
if (exist('bias','var') && (bias == 1))
    X = [X ones(obs_count,1)];
end
obs_dim = size(X,2);
if exist('wi','var')
    w = wi;
else
    w = zeros(obs_dim,1);
end
if (numel(w) ~= obs_dim)
    error('Invalid weight vector: wrong number of weights.\n');
end
if ~exist('max_iter','var')
    max_iter = 500;
end

% Compute the objective at initialization point
f = funobj_wl1_logreg(w, X, Y, lam);
%fprintf('STARTING OBJ: %.4f\n', f);

% Setup options for minFunc
options = struct();
options.Display = 'off';
options.Method = 'lbfgs';
options.Corr = 5;
options.LS = 1;
options.LS_init = 3;
options.MaxIter = max_iter;
options.MaxFunEvals = 2*max_iter;
options.TolX = 1e-8;

funObj = @( b ) funobj_wl1_logreg(b, X, Y, lam);
w = minFunc(funObj, w, options);

return

end

function [ obj dW ] = funobj_wl1_logreg(w, X, Y, lam)
% Compute the objective function value for pseudo-l1-regularized weighted
% logistic regression. This uses a weighted binomial deviance loss:
%     l(x, y, w, lam) = log(1 + exp(w'*x)) + (lam/obs_dim)*sum(sqrt(w.^2+eps)))
%
% Parameters:
%   w: parameter values for which to compute objective and gradient
%   X: input observations
%   Y: weighted classifications
%   lam: regularization weight
%
% Output:
%   obj: objective value
%   dW: gradient of objective with respect to parameter values in w
%

obs_count = size(X,1);
obs_dim = size(X,2);
lam = lam * (obs_count / obs_dim);

w_s = sqrt(w.^2 + 1e-8);

% Decompose Y into sign and magnitude components
Ys = sign(Y);
Ym = abs(Y);
% Compute objective function value
F = X*w;
L = Ym .* log(exp(-Ys.*F) + 1);
loss_spars = sum(w_s);
loss_class = sum(L);
obj = loss_class + (lam * loss_spars);

if (nargout > 1)
    % Compute objective function gradients
    dS = w ./ w_s;
    % Loss gradient with respect to output at each input observation
    dL = Ym .* (-Ys .* (exp(-Ys.*F) ./ (exp(-Ys.*F) + 1)));
    % Backpropagate through input observations
    dC = sum(bsxfun(@times, X, dL));
    dW = dC' + (lam * dS);
end

return

end

