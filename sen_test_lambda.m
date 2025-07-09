%%************************************************************************
%% Run random matrix completion problems.
%%************************************************************************

clear;clc;%close all; 
restoredefaultpath;
addpath(genpath('solvers'));
addpath(genpath('PROPACKmod'));

%% Generate random test problem
scenario = 'noisy';
errtype = 5;   % 2, 5   %% 1-5
ntest = 1;

OPTIONS_SPC = struct('maxiter', 500, 'printyes', 0, 'tol', 1.0e-5);
OPTIONS_AMM = struct('maxiter', 500, 'printyes', 0, 'tol', 5.0e-5);

%% Initialization for test problem
nr = 1000; nc = 1000; % nr=3000, nr=5000
rstar = 10;
d1 = 5 * rstar; d2 = 5 * rstar;
SR = 0.1; % [0.02  0.05  0.1  0.15  0.2  0.25  0.3  0.35  0.4     0.45     0.5   ];
nlamb = 20;
lamlen = logspace(-1,3,nlamb)';
lamlen_SPC = logspace(-1,3,nlamb)';
ns = length(lamlen); 
% lamlen_SPC = [0  0.001  0.01  0.1  1     2     3    4     5    6    7   8   10 ];
% lamlen =  [0   0.001  0.002   0.004  0.005  0.006  0.008    0.01  0.012  0.015  0.018   0.02  0.03  ];

%% Initialize result matrices
relerr_PAM = ones(ntest, ns);
rankX_PAM = zeros(ntest, ns);
CPUtime_PAM = zeros(ntest, ns);

relerr_SPC = ones(ntest, ns);
RankX_SPC = zeros(ntest, ns);
CPUtime_SPC = zeros(ntest, ns);

%% Main loop
for i = 1:length(lamlen)
    lami_SPC = lamlen_SPC(i);
    lami = lamlen(i);
    
    for t = 1:ntest
        fprintf('\nTest %d of %d\n', t, ntest);
        
        randstate = 100 * i * t;
        rng(randstate); % Set random seed
        
        if strcmp(scenario, 'noiseless')
            noiseratio = 0;
        else
            noiseratio = 0.1;
        end
        
        fprintf('\n nr = %2.0d, nc = %2.0d, rank = %2.0d\n', nr, nc, rstar);
        
        p = round(SR * nr * nc); % Number of sampled entries
        
        %% Generate the true matrix
        A = randn(nr, d1);
        B = randn(nc, d2);
        Xstar = randn(d1, rstar) * randn(d2, rstar)';
        Xstar = Xstar / norm(Xstar, 'fro');
        Mstar = A * Xstar * B';
        normM = norm(Mstar, 'fro');
        
        %% Uniform sampling
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
        
        %% Initialization
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
        
        %% SInf_AMM_CutCol
        tstart = tic;
        options = 1.0e-6;
        AMB = (A\Mhat)/B'; % A' * Mhat * B;

        [U, dA, V] = lansvd(AMB, r, 'L', options);
        dA = diag(dA)';
        Ustart = U(:, 1:r) .* dA(1:r).^(1/2);%.^(1/4)
        Vstart = V(:, 1:r) .* dA(1:r).^(1/2);%.^(1/4)
        
%         Ustart = randn(d1, r);
%         Vstart = randn(d2, r);
        Pstart = Ustart; %orth(Ustart);
        Qstart = Vstart; %orth(Vstart);
        dstart = ones(1, r);
        
        OPTIONS_SPC.Lip_const = .1 * ABnorm;
        lambda = lami_SPC*normb; % *1.0e5*
%         SInf_rankX = 0;
%         AXBopt = 0;
        [SInf_rankX, AXBopt] = SInf_AMM_CutCol_new(Mstar, bb, Pstart, Qstart, dstart, ...
            nzidx, A, B, lambda, pars, OPTIONS_SPC);
        ttime = toc(tstart);
        diffM = AXBopt - Mstar;
        relerr = min(norm(diffM(nzidx), 'fro') / normM, 1);
        relerr_SPC(t, i) = relerr;
        RankX_SPC(t, i) = SInf_rankX;
        CPUtime_SPC(t, i) = ttime;
        
        %% AMM_solver
        tstart = tic;
        [U, dA, V] = lansvd(AMB, r, 'L', options);
        dA = diag(dA)';
        max_dA = max(dA);
        Ustart = U(:, 1:r) .* dA(1:r).^(1/2);%.^(1/4)
        Vstart = V(:, 1:r) .* dA(1:r).^(1/2);%.^(1/4)
        
        OPTIONS_AMM.LipX = 0.03*ABnorm;
        lambda = lami*normb; % *1.0e3

%         AXBopt = 0;
%         rankXopt = 0;
%         pause(2);
        [AXBopt, rankXopt, outs] = AMM_Fnorm(Mstar,bb, Ustart, Vstart, nzidx, A, B, OPTIONS_AMM, pars, lambda);
%         [AXBopt, rankXopt, outs] = my_IMC(Mstar,bb, Pstart, Qstart, nzidx, A, B, OPTIONS_AMM, pars, lambda);
%         [AXBopt, rankXopt, outs] = AMM_solver1(Mstar, bb, Ustart, Vstart, nzidx, A, B, OPTIONS_AMM, pars, lambda);
        time = toc(tstart);
        diffM = AXBopt - Mstar;
        relerr = min(norm(diffM(nzidx), 'fro') / normM,1);
        if rankXopt <= 1
            rankXopt = rankX_PAM(t, i-1);
            relerr = relerr_PAM(t, i-1);
            time = CPUtime_PAM(t, i-1);
        end
        rankX_PAM(t, i) = rankXopt;
        relerr_PAM(t, i) = relerr;
        CPUtime_PAM(t, i) = time;
    end
end

%% Save results
results = struct(...
    'lambda', lamlen,...
    'matRE_SPC', relerr_SPC, ...
    'matRankX_SPC', RankX_SPC, ...
    'mattime_SPC', CPUtime_SPC, ...
    'matrank', rankX_PAM, ...
    'matrelerr', relerr_PAM, ...
    'mattime', CPUtime_PAM);

save(sprintf('results_nc%d_nr%d_r%d_SR%1.2f.mat',nc,nr,rstar,SR), 'results');
results = struct(...
    'lambda', lamlen,...
    'relerr_SPC', mean(relerr_SPC,1)', ...
    'RankX_SPC', mean(RankX_SPC,1)', ...
    'CPUtime_SPC', mean(CPUtime_SPC,1)', ...
    'relerr_PAM', mean(relerr_PAM,1)', ...
    'rankX_PAM', mean(rankX_PAM,1)', ...
    'CPUtime_PAM', mean(CPUtime_PAM,1)');
resultsTable = struct2table(results);
%% Display average results
% results = struct(...
%     'lambda', lamlen,...
%     'matRE_SPC', relerr_SPC, ...
%     'matRankX_SPC', RankX_SPC, ...
%     'mattime_SPC', CPUtime_SPC, ...
%     'matrank', rankX_PAM, ...
%     'matrelerr', relerr_PAM, ...
%     'mattime', CPUtime_PAM);

% lamlen = results.lambda;
% relerr_SPC = mean(results.matRE_SPC, 1);
% RankX_SPC = mean(results.matRankX_SPC, 1);
% CPUtime_SPC = mean(results.mattime_SPC, 1);
% relerr_PAM = mean(results.matrelerr, 1);
% rankX_PAM = mean(results.matrank, 1);
% CPUtime_PAM = mean(results.mattime, 1);
  
relerr_SPC = results.relerr_SPC;
RankX_SPC = results.RankX_SPC;
CPUtime_SPC = results.CPUtime_SPC;
relerr_PAM = results.relerr_PAM;
rankX_PAM = results.rankX_PAM;
CPUtime_PAM = results.CPUtime_PAM;
%% close all;
figure;%('Position', [300, 300, 1100, 350]);

% % Subplot 1: lambda vs relerr
% subplot(1, 3, 1);
% yyaxis left;
% loglog(lamlen, relerr_PAM, 'bo-', 'LineWidth', 2, 'MarkerSize', 8); hold on;
% yyaxis right;
% semilogx(lamlen, rankX_PAM, 'gs-', 'LineWidth', 2, 'MarkerSize', 8);
% xlabel('Lambda');
% ylabel('Relative Error');
% title('Lambda vs PAM');
% legend('Relative Error', 'Rank of X', 'Location', 'best');
% grid on;
% 
% % Subplot 2: lambda vs CPUtime
% subplot(1, 3, 2);
% yyaxis left;
% loglog(lamlen, relerr_SPC, 'bo-', 'LineWidth', 2, 'MarkerSize', 8); hold on;
% yyaxis right;
% semilogx(lamlen, RankX_SPC, 'gs-', 'LineWidth', 2, 'MarkerSize', 8);
% xlabel('Lambda');
% ylabel('Rank of X');
% title('Lambda vs SPC');
% legend('Relative Error', 'Rank of X', 'Location', 'best');
% grid on;
% 
% % Subplot 3: lambda vs RankX
% subplot(1, 3, 3);
% semilogx(lamlen, CPUtime_SPC, 'bo-', 'LineWidth', 2, 'MarkerSize', 8); hold on;
% semilogx(lamlen, CPUtime_PAM, 'gs-', 'LineWidth', 2, 'MarkerSize', 8);
% xlabel('Lambda');
% ylabel('CPU Time (seconds)');
% title('Lambda vs CPU Time');
% legend('SPC', 'PAM', 'Location', 'best');
% grid on;
Ran = rstar;
sr_PAM = find(rankX_PAM == Ran);
sr_SPC = find(RankX_SPC == Ran);

% % Subplot 1: lambda vs relerr
subplot(1, 3, 2);
loglog(lamlen, relerr_SPC, 'bo-', 'LineWidth', 2, 'MarkerSize', 8); hold on;
loglog(lamlen, relerr_PAM, 'gs-', 'LineWidth', 2, 'MarkerSize', 8);
loglog(lamlen(1), RankX_SPC(1), 'r', 'LineWidth', 2, 'MarkerSize', 8);
loglog(lamlen(sr_SPC), relerr_SPC(sr_SPC), 'ro-', 'LineWidth', 2, 'MarkerSize', 8);
loglog(lamlen(sr_PAM), relerr_PAM(sr_PAM), 'rs-', 'LineWidth', 2, 'MarkerSize', 8);

xlabel('$\lambda$', 'Interpreter', 'latex');
ylabel('Relative Error');
title('(b) $\lambda$ vs Relative Error', 'Interpreter', 'latex');  
legend({'Algorithm 1', 'Algorithm 2', 'True Rank'}, ...
    'Location', 'Northwest');
grid on;

% Subplot 2: lambda vs RankX
subplot(1, 3, 1);
semilogx(lamlen, RankX_SPC, 'bo-', 'LineWidth', 2, 'MarkerSize', 8); hold on;
semilogx(lamlen, rankX_PAM, 'gs-', 'LineWidth', 2, 'MarkerSize', 8);
loglog(lamlen(1), RankX_SPC(1), 'r', 'LineWidth', 2, 'MarkerSize', 8);
loglog(lamlen(sr_SPC), RankX_SPC(sr_SPC), 'ro-', 'LineWidth', 2, 'MarkerSize', 8);
loglog(lamlen(sr_PAM), rankX_PAM(sr_PAM), 'rs-', 'LineWidth', 2, 'MarkerSize', 8);

xlabel('$\lambda$', 'Interpreter', 'latex');
ylabel('Rank of X');
title('(a) $\lambda$ vs Rank of X', 'Interpreter', 'latex');
legend({'Algorithm 1', 'Algorithm 2', 'True Rank'}, ...
    'Location', 'best');
grid on;

% Subplot 3: lambda vs CPUtime
subplot(1, 3, 3);
semilogx(lamlen, CPUtime_SPC, 'bo-', 'LineWidth', 2, 'MarkerSize', 8); hold on;
semilogx(lamlen, CPUtime_PAM, 'gs-', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('$\lambda$', 'Interpreter', 'latex');
ylabel('CPU Time (s)');
title('(c) $\lambda$ vs CPU Time', 'Interpreter', 'latex');
legend({'Algorithm 1', 'Algorithm 2'},  'Location', 'Northeast');
grid on;

% Adjust layout for better spacingcl
set(gcf, 'Position', [300, 300, 1200, 350]); % Resize the figure
set(gcf,'Units','Inches');
pos = get(gcf,'Position');
set(gcf,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',...
    [pos(3), pos(4)]);
print(gcf,sprintf('fig_nc%d_nr%d_r%d_SR%1.2f.pdf',nc,nr,rstar,SR),'-dpdf','-r0');

% sgtitle('Performance Comparison of SPC and PAM Algorithms', ...
%     'FontSize', 14, 'FontWeight', 'bold');

% [EOF]