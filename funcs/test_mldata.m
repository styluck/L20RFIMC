function [re_pammse, re_pamm] = test_mldata(sr, exp_times, name, isSaving, isDisp)

    %% Initialization
    p_num = numel(sr);

    % Preallocate result matrices
    re_pamm = zeros(p_num, exp_times);
    re_pammse = zeros(p_num, exp_times);
    time_pamm = zeros(p_num, exp_times); % Matrix to store computational times for PAMM
    time_pammsc = zeros(p_num, exp_times); % Matrix to store computational times for PAMM_SC

    % Load and preprocess the data
    data = load([name, '.mat']);
    Mdata = data.Mdata;
    A = data.B;

    % Reduce data size for speed
    Mdata = Mdata(:, 1:5000)';
    A = A(1:5000, :);
    lambda = .5;
    rstar = 5;
    [nr, nc] = size(Mdata);
    B = eye(nc);
    d1 = size(A,2);
    d2 = nc;
    r = min(min(d1, d2), 100);
    Xstar = randn(d1, rstar) * randn(d2, rstar)';
    Xstar = Xstar / norm(Xstar, 'fro');
    Mstar = A * Xstar * B';
    normM = norm(Mstar, 'fro');
    ABnorm = (norm(A, 2) * norm(B, 2))^2;
    % Identify positive and negative entries
    positive_index = find(Mdata(:) > 0);
    positive_number = length(positive_index);
    negative_index = find(Mdata(:) <= 0);
    negative_number = length(negative_index);

    OPTIONS_SPC = struct('maxiter', 500, 'printyes', 1, 'tol', 1.0e-5);
    OPTIONS_AMM = struct('maxiter', 500, 'printyes', 1, 'tol', 1.0e-5);
    %% Main Loop
    for i = 1:p_num
        percent = sr(i);
        for j = 1:exp_times
            % Generate random observed entries
            Omega = generate_random_observations(positive_index, positive_number, ...
                negative_index, negative_number, percent, nr, nc);
            Omega_linear = find(Omega);
            
            bb = Mdata(Omega_linear);
            Mhat = zeros(nr, nc);
            Mhat(Omega_linear) = bb;
            
            p = percent*nr*nc;
            normb = norm(bb);
            pars = struct('normM', normM, ...
            'normb', normb, ...
            'nc', nc, ...
            'nr', nr, ...
            'd1', d1, ...
            'd2', d2, ...
            'p', p, ...
            'r', r);
            if isDisp
                fprintf('Sampling rate = %f, Test #%d \n', percent, j);
            end

            %% PAMM Solver
            tstart = tic;

%             pars = initialize_solver_parameters( ...
%                 nr, nc, d1, nc, r, Mdata, Omega_linear, Mdata(Omega_linear), lambda); % Encapsulated parameters

            options = 1.0e-6;
            AMB = A' * Mhat * B;
            [U, dA, V] = lansvd(AMB, r, 'L', options);
            if length(dA) < r
                r = length(dA);
                pars.r = r;
            end
            dA = diag(dA)';
            max_dA = max(dA);
            Ustart = U(:, 1:r) .* dA(1:r).^(1/4);
            Vstart = V(:, 1:r) .* dA(1:r).^(1/4);
                
            OPTIONS_AMM.Lip_const = 0.01 * max_dA^(1/2) * ABnorm;

            [AXBopt, rankXopt, outs] = AMM_solver1(Mdata, bb, Ustart, Vstart, ...
                Omega_linear, A, B, OPTIONS_AMM, pars, lambda);
            timeElapsed = toc(tstart);
%             AXBopt = A * Xopt * B';
            relErr = norm(AXBopt - Mdata, 'fro') / norm(Mdata, 'fro');
            re_pamm(i, j) = relErr;
            time_pamm(i, j) = timeElapsed; % Store PAMM computational time

            %% SInf_AMM_CutCol Solver
            tstart = tic;
            Ustart = randn(d1, rstar);
            Vstart = randn(d2, rstar);
            Pstart = orth(Ustart);
            Qstart = orth(Vstart);
            dstart = ones(1, rstar);
            
            OPTIONS_SPC.Lip_const = 0.1 * ABnorm;
%             [Xopt, rankXopt, outs] = SInf_AMM_CutCol1(Mdata, Omega_linear, rstar, A, B, pars);
            [rankXopt, AXBopt] = SInf_AMM_CutCol1(Mdata, bb, Pstart, Qstart, dstart, ...
                Omega_linear, A, B, lambda, pars, OPTIONS_SPC);
            timeElapsed = toc(tstart);
%             AXBopt = A * Xopt * B';
            relErr = norm(AXBopt - Mdata, 'fro') / norm(Mdata, 'fro');
            re_pammse(i, j) = relErr;
            time_pammsc(i, j) = timeElapsed; % Store PAMM_SC computational time
        end
    end

    %% Plot Results

    % Sampling Rate vs Relative Error
    figure('numbertitle', 'off', 'name', 'Sampling Rate vs Relative Error and computational time');
    subplot(1,2,1)
    plot(sr, mean(re_pammse, 2), '-.k', 'MarkerSize', 8, 'LineWidth', 1); % PAMM-sc
    hold on;
    plot(sr, mean(re_pamm, 2), '-m', 'MarkerSize', 8, 'LineWidth', 1); % PAMM
    xlim([sr(1), sr(end)]);
    ylim([0 1]);
    legend('PAMM\_SC', 'PAMM', 'Location', 'SouthEast');
    xlabel('Sampling Rate', 'Interpreter', 'latex', 'FontSize', 12);
    ylabel('Relative Error', 'FontSize', 12);

    % Save the Sampling Rate vs Relative Error figure
    if isSaving
        timestamp = datestr(datetime('now'), 'yyyy-mm-dd_HH-MM-SS');
        plotFile = ['figures\', name, '_', timestamp, '_rel_error.png'];
        saveas(gcf, plotFile);
    end

    % Sampling Rate vs Computational Time
    subplot(1,2,2)
    plot(sr, mean(time_pamm, 2), '-.r', 'MarkerSize', 8, 'LineWidth', 1); % PAMM time
    hold on;
    plot(sr, mean(time_pammsc, 2), '-b', 'MarkerSize', 8, 'LineWidth', 1); % PAMM_SC time
    xlim([sr(1), sr(end)]);
    ylim([0, max([mean(time_pamm, 2); mean(time_pammsc, 2)]) * 1.2]);
    legend('PAMM', 'PAMM\_SC', 'Location', 'NorthWest');
    xlabel('Sampling Rate', 'Interpreter', 'latex', 'FontSize', 12);
    ylabel('Computational Time (s)', 'FontSize', 12);

    % Save the Sampling Rate vs Computational Time figure
    if isSaving
        timestamp = datestr(datetime('now'), 'yyyy-mm-dd_HH-MM-SS');
        timePlotFile = ['figures\', name, '_', timestamp, '_comp_time.png'];
        saveas(gcf, timePlotFile);
    end

    %% Save Results
    if isSaving
        timestamp = datestr(datetime('now'), 'yyyy-mm-dd_HH-MM-SS');
        saveFile = ['savings\', name, '_', timestamp, '.mat'];
        save(saveFile, 'exp_times', 'sr', 're_pamm', 're_pammse', 'time_pamm', 'time_pammsc');
        fprintf('Results saved to: %s\n', saveFile);
    end
end

%% Helper Functions

% Generate random observation indices
function Omega = generate_random_observations(pos_idx, pos_num, neg_idx, neg_num, percent, m, n)
    Omega = zeros(m, n);
    pos_rand = randperm(pos_num);
    pos_select = pos_idx(pos_rand(1:ceil(pos_num * percent)));
    Omega(pos_select) = 1;

    neg_rand = randperm(neg_num);
    neg_select = neg_idx(neg_rand(1:ceil(neg_num * percent)));
    Omega(neg_select) = 1;
end
