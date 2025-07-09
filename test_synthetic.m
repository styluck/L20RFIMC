clear; close all; clc;
restoredefaultpath;
addpath(genpath('solvers'));
addpath(genpath('sgimc'));
addpath(genpath('PROPACKmod'));

% Initialize result storage
resultsTable = table();
%% Parameters
scenario = 'noisy'; % 'noiseless' or 'noisy'
errtype = 5; % 2, 5 (1-5)
ntest = 5; % Number of tests per configuration
SR = 0.2; % Sampling ratio
ll = 50;
lamlen = logspace(-1, 3, ll)'; % Lambda values
 
% Define ranges for nr, nc, and rstar
ranges = [
    500     500;
    1000	2000;
    1000	3000;
    2000	1000;
    2000	2000;
    2000	3000;
    3000	3000;
    3000	5000;
    5000	5000];
nrange = size(ranges,1);
rstar_range = [10 20]; % , 20

% Define the algorithms to test
algorithms = {'SInf_AMM_CutCol1', 'AMM_Fnorm', 'GNIMC', 'SGIMC'};%'SGIMC', }; % Add more algorithms here
numAlgorithms = length(algorithms);

params = struct();
params.eta = 1.0;                      % ADMM penalty
params.maxOuterIter = 10;              % outer loops
params.maxInnerIter = 5;              % ADMM inner steps
params.rtol = 1e-5;
params.atol = 1e-8;
params.loss = 'squared';
params.verbose = false;

%% Main loop over all combinations of nr, nc, and rstar
ranges1 = ranges(:,1);
ranges2 = ranges(:,2);
% lams1 = lams(:,1);
% lams2 = lams(:,2);
parfor ni = 1:nrange
    nc = ranges1(ni);
    nr = ranges2(ni);
    for rstar = rstar_range
        fprintf('\nTesting nr = %d, nc = %d, rstar = %d\n', nr, nc, rstar);

        % Initialize result matrices for this configuration
        relerr = cell(numAlgorithms, 1);
        rankX = cell(numAlgorithms, 1);
        CPUtime = cell(numAlgorithms, 1);

        for k = 1:numAlgorithms
            relerr{k} = zeros(ntest, length(lamlen));
            rankX{k} = zeros(ntest, length(lamlen));
            CPUtime{k} = zeros(ntest, length(lamlen));
        end

        % Test loop
        for i = 1:length(lamlen)
            lami = lamlen(i);
%             lami1 = lams1(ni);
%             lami2 = lams2(ni);

            for t = 1:ntest
                fprintf('\nTest %d of %d\n', t, ntest);

                randstate = 100 * i * t;
                rng(randstate); % Set random seed

                if strcmp(scenario, 'noiseless')
                    noiseratio = 0;
                else
                    noiseratio = 0.1;
                end

                % Generate the true matrix
                d1 = 5 * rstar; d2 = 5 * rstar;
                A = randn(nr, d1);
                B = randn(nc, d2);
                Xstar = randn(d1, rstar) * randn(d2, rstar)';
                Xstar = Xstar / norm(Xstar, 'fro');
                Mstar = A * Xstar * B';
                normM = norm(Mstar, 'fro');

                % Uniform sampling
                p = round(SR * nr * nc);
                ind = randperm(nr * nc);
                nzidx = ind(1:p)';
                bb = Mstar(nzidx);

                if strcmp(scenario, 'noiseless')
                    xi = sparse(p, 1);
                    sigma = 0;
                else
                    randnvec = randn(p, 1);
                    sigma = noiseratio * norm(bb) / norm(randnvec);
                    xi = sigma * randnvec;
                end

                bb = bb + xi;
                Mhat = zeros(nr, nc);
                Mhat(nzidx) = bb;

                % Initialization
                normb = norm(bb);
                r = min(min(d1, d2), 100);
                pars = struct('normM', normM, ...
                    'normb', normb, ...
                    'nc', nc, ...
                    'nr', nr, ...
                    'd1', d1, ...
                    'd2', d2, ...
                    'p', p, ...
                    'r', r);

                ABnorm = (norm(A, 2) * norm(B, 2))^2;

                % Loop over algorithms% Loop over algorithms
                for k = 1:numAlgorithms
                    algorithm = algorithms{k};
                    tstart = tic;

                    if strcmp(algorithm, 'SInf_AMM_CutCol1')
                        Ustart = randn(d1, r);
                        Vstart = randn(d2, r);
                        Pstart = orth(Ustart);
                        Qstart = orth(Vstart);
                        dstart = ones(1, r);

                        OPTIONS_SPC = struct('maxiter', 1000, 'printyes', 0, ...
                            'tol', 1.0e-5, 'Lip_const', 0.8 * ABnorm);
                        lambda = lami1 * normb;

                        [SInf_rankX, AXBopt] = SInf_AMM_CutCol_new(Mstar, bb, Pstart, Qstart, dstart, ...
                            nzidx, A, B, lambda, pars, OPTIONS_SPC);
                        rankXopt = SInf_rankX;
                    elseif strcmp(algorithm, 'AMM_Fnorm')
                        tstart = tic;
                        options = 1.0e-6;
                        AMB = (A\Mhat)/B'; % A' * Mhat * B;
                        [U, dA, V] = lansvd(AMB, r, 'L', options);
                        dA = diag(dA)';
                        max_dA = max(dA);
                        Ustart = U(:, 1:r) .* dA(1:r).^(1/2);%.^(1/4)
                        Vstart = V(:, 1:r) .* dA(1:r).^(1/2);%.^(1/4)

                        OPTIONS_AMM = struct('maxiter', 1000, 'printyes', 0, ...
                            'tol', 5.0e-5, 'LipX', 0.03 * ABnorm);
                        lambda = lami2 * normb;

                        [AXBopt, rankXopt, outs] = AMM_Fnorm(Mstar,bb, Ustart, Vstart, ...
                            nzidx, A, B, OPTIONS_AMM, pars, lambda);
                    elseif strcmp(algorithm, 'SGIMC')
                        % ---- SGIMC parameters ----
                        % ---- observed matrix R ----
                        R = zeros(size(Mstar));
                        R(nzidx) = bb;

                        % ---- run SGIMC ----
                        [Wsg, Hsg, ~] = sgimc_qaadmm(A, B, R, r, params, lami*normb);
                        AXBopt = (A * Wsg) * (B * Hsg)';       % reconstructed matrix
                        rankXopt = rank(AXBopt);               % numerical rank

                    elseif strcmp(algorithm, 'GNIMC')  
                        R(nzidx) = bb;
  
                        [AXBopt, ~, ~, ~] = GNIMC(Mstar, nzidx, r, A, B, opts_GNIMC); 
                        rankXopt = rank(AXBopt);               % numerical rank
 
                    else
                        % Add more algorithm cases here
                        error(['Algorithm ' algorithm ' is not implemented.']);
                    end
                    time = toc(tstart);
                    relerr{k}(t, i) = norm(AXBopt - Mstar, 'fro') / normM;
                    rankX{k}(t, i) = rankXopt;
                    CPUtime{k}(t, i) = time;
                end
            end
        end

        % Store results for this configuration
        results = struct('nr', ones(ll,1)*nr, ...
            'nc', ones(ll,1)*nc, ...
            'rstar', ones(ll,1)*rstar, ...
            'lambda', lamlen);

        for k = 1:numAlgorithms
            algorithm = algorithms{k};
            results.(['relerr_' algorithm]) = mean(relerr{k}, 1)';
            results.(['rankX_' algorithm]) = mean(rankX{k}, 1)';
            results.(['CPUtime_' algorithm]) = mean(CPUtime{k}, 1)';
        end

        % Append results to the table
        resultsTable = [resultsTable; struct2table(results)];
    end
end

%% Save results
save(['results_all_SR' num2str(SR) '.mat'], 'resultsTable');
writetable(resultsTable, ['results_all_SR' num2str(SR) '.csv']);