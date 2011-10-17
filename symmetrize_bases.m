function [ B_out ] = symmetrize_bases( B )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

B_out = zeros(size(B));
for  i=1:size(B,3),
    B_out(:,:,i) = (squeeze(B(:,:,i)) + squeeze(B(:,:,i))') ./ 2.0;
end

return

end

