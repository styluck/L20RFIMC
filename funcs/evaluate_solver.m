function results = evaluate_solver(solver_name, Mstar, nzidx, A, B, r, pars, results, row_idx)
    % Evaluate the specified solver and store results in the table
    tstart = tic;
    [Xopt, rankXopt, ~] = feval(solver_name, Mstar, nzidx, r, A, B, pars);
    ttime = toc(tstart);

    % Compute metrics
    AXBopt = A * Xopt * B';
    relerr = norm(AXBopt - Mstar, 'fro') / pars.normM;

    % Store results in the table
    results.RelativeError(row_idx) = relerr;
    results.Rank(row_idx) = rankXopt;
    results.Time(row_idx) = ttime;
end