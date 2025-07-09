% function [M, A, B, U, V] = generate_matrix(n1,n2,d1,d2,singular_values)
% %
% % INPUT:  n1,n2 = number of rows and columns in X
% %         d1,d2 = number of columns in A,B
% %         singular_values = list of non-zero singular values
%
% % OUTPUT: X = n1 x n2 matrix of rank r
% %         A = n1 x d1 left singular vectors
% %         B = n2 x d2 right singular vectors
%
% r = length(singular_values);
%
% Z = randn(n1, d1);
% [A, ~, ~] = svd(Z,'econ');
% Z = randn(n2, d2);
% [B, ~, ~] = svd(Z,'econ');
% Z = randn(d1, r);
% [U, ~, ~] = svd(Z,'econ');
% Z = randn(d2, r);
% [V, ~, ~] = svd(Z,'econ');
% D = diag(singular_values);
% U = U*D;
% M = A * U *  V' * B';

function [Mstar, A, B, nzidx, bb, normM] = generate_matrix(nr, nc, d1, d2, r, SR, sample_type, scenario, noiseratio)
% Generate random true matrix and sampled entries

A = randn(nr, d1);
B = randn(nc, d2);
X = randn(d1, r) * randn(d2, r)';
Mstar = A * X * B';
normM = norm(Mstar, 'fro');
p = round(SR * nr * nc);

% Sample indices based on type
idx = randperm(nr * nc);
nzidx = idx(1:p)';
bb = Mstar(nzidx);

% Sampling logic
switch sample_type
    case 'uniform'
        fprintf('\n *********** Uniform Sampling ***************\n');
        fprintf('Sampling Ratio (SR): %2.2f, Noise Ratio: %2.2f\n', SR, noiseratio);

        % Uniform sampling
        [nzidx, zidx] = performUniformSampling(nr, nc, p);

    case 'nonuniform'
        fprintf('\n *********** Non-Uniform Sampling ***********\n');

        % Non-uniform sampling
        [nzidx, zidx] = performNonUniformSampling(nr, nc, p);

    otherwise
        error('Invalid sample type: %s. Choose "uniform" or "nonuniform".', sample_type);
end


% Add noise if necessary
if strcmp(scenario, 'noisy')
    randnvec = randn(p,1);
    sigma = noiseratio * norm(bb) / norm(randnvec);
    bb = bb + sigma * randnvec;
end
end

% Function for uniform sampling
function [nzidx, zidx] = performUniformSampling(nr, nc, p)
    totalEntries = nr * nc;
    idx = randperm(totalEntries);
    nzidx = idx(1:p)';        % Indices for sampled entries
    zidx = idx(p+1:end)';     % Indices for unsampled entries
end

% Function for non-uniform sampling
function [nzidx, zidx] = performNonUniformSampling(nr, nc, p)
    % Generate row and column probability vectors
    pvec = createProbabilityVector(nr);
    qvec = createProbabilityVector(nc);

    % Compute entry-wise probabilities
    probmatrix = rand(nr, nc) .* (pvec * qvec');

    % Sort probabilities in descending order
    [probvec, sortidx] = sort(probmatrix(:), 'descend');
    nzidx = sortidx(1:p);            % Indices for sampled entries
    zidx = sortidx(p+1:end);         % Indices for unsampled entries
end

% Helper function to create a probability vector for non-uniform sampling
function probvec = createProbabilityVector(dim)
    probvec = ones(dim, 1);
    cnt = round(0.1 * dim);
    probvec(1:cnt) = 2 * probvec(1:cnt);
    probvec(cnt+(1:cnt)) = 4 * probvec(cnt+(1:cnt));
    probvec = dim * probvec / sum(probvec); % Normalize
end
