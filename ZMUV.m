function [ Xo ] = ZMUV( Xi )
% Set the columns of Xi to zero-mean, unit-variance

Xo = zeros(size(Xi));
for i=1:size(Xi,2),
    x = Xi(:,i);
    Xo(:,i) = (x - mean(x)) ./ std(x);
end

return

end

