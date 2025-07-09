function [H,omega,omega_2d] = generate_mask(n1,n2,m)
%
%%% code adapted from MatrixIRLS by Christian Kuemmerle %%%
%
% INPUT:    n1 = number of rows
%           n2 = number of columns
%           nv = number of nonzero entries
%
% OUTPUT:   H = n1 x n2 matrix of zeros and ones
%           omega = m-vector with linear indices of non-zero entries in H
%           omega_2d = 2d version of omega

% obs_ratio = 0.3;
% Omega = rand(n, m) < obs_ratio;
% P_Omega = @(X) X .* Omega;
% M_obs = P_Omega(m + 0.1 * randn(n, m));

omega = (sort(randperm(n1*n2,m)))';
i_Omega = mod(omega,n1);
i_Omega(i_Omega==0) = n1;
j_Omega = floor((omega-1)/n1)+1;
H = sparse(i_Omega,j_Omega,ones(m,1),n1,n2);
omega_2d = zeros(m,2);
[omega_2d(:,1), omega_2d(:,2)] = ind2sub([n1,n2],omega);

end
