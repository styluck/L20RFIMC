% ===================== Demo: SGIMC Benchmarking ========================
% Setup: Compare SGIMC's performance with your existing metrics
% Author: Adapted by ChatGPT, July 2025
% =======================================================================

% === 1. Generate or load test matrix Mstar ===========================
% Mstar: ground-truth full matrix
n1 = 500;
n2 = 400;
r_true = 10;
Utrue = randn(n1, r_true); Vtrue = randn(n2, r_true);
Mstar = Utrue * Vtrue';    % ground-truth full matrix
normM = norm(Mstar, 'fro');

% === 2. Create mask and observed entries =============================
p_obs = 0.2;                     % percent observed
nzidx = rand(n1, n2) < p_obs;    % observed indices
R = zeros(n1, n2); R(nzidx) = Mstar(nzidx);   % sparse observed matrix

% === 3. Feature matrices A and B (X and Y) ===========================
d1 = 60; d2 = 50;                % feature dimensions
A = randn(n1, d1); B = randn(n2, d2);

% === 4. Set SGIMC parameters =========================================
params.lambda_l1 = 1.0;
params.lambda_group = 0.1;
params.lambda_ridge = 1.0;
params.eta = 1.0;
params.maxOuterIter = 50;
params.maxInnerIter = 20;
params.rtol = 1e-5;
params.atol = 1e-8;
params.loss = 'squared';
params.verbose = true;

% === 5. Run SGIMC ====================================================
tstart = tic;
[W, H, hist] = sgimc_qaadmm(A, B, R, r_true, params);
ttime = toc(tstart);

% === 6. Recover completed matrix =====================================
Mhat = (A * W) * (B * H)';        % predicted full matrix
diffM = Mhat - Mstar;
relerr = min(norm(diffM(nzidx), 'fro') / normM, 1);

% === 7. Output performance ===========================================
fprintf('SGIMC: rel.err = %.4e, rank = %d, time = %.2fs\n', ...
    relerr, rank(Mhat), ttime);

% === 8. Store for comparison if looping ==============================
% relerr_SGIMC(t, i) = relerr;
% RankX_SGIMC(t, i) = rank(Mhat);
% CPUtime_SGIMC(t, i) = ttime;
