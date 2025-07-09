function display_results(results)
    % Display summary results for each algorithm and lambda
    summary = groupsummary(results, {'Algorithm', 'Lambda'}, {'mean'}, ...
        {'RelativeError', 'Time', 'Rank'});
    disp(summary);
end
