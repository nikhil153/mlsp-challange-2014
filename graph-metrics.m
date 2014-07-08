%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Methods taken from :
% Functional Connectivity and Brain Networks in Schizophrenia
% Lynall et al 2010 Journal of Neuroscience
%
% Depends on the Brain Connectivity Toolbox (BCT)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%filename prefix -- train or test
filetype = 'test';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Import Data and Format to a Set of Correlation Matricies
FNC = dlmread([ filetype '_FNC.csv'], ',', 1, 1);
%SBM = dlmread('train_SBM.csv', ',', 1, 1);

% import mappings
FNC_map = dlmread('rs_fMRI_FNC_mapping.csv', ',', 1, 1);

dims = size(FNC);
n_subj = dims(1);

% linear map values
count = 1;
for label = unique(FNC_map)';
    idx = find(FNC_map == label);
    FNC_map(idx) = count;
    count = count+1;
end

% create subject wise graphs
cmat = zeros(28, 28, n_subj);
for subj = 1:n_subj;
    for c = 1:378;
        
        % get index
        x = FNC_map(c, 1);
        y = FNC_map(c, 2);

        cmat(x, y, subj) = FNC(subj, c);
        cmat(y, x, subj) = FNC(subj, c);
    end
end

% clear RAM
clearvars FNC FNC_map

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate Graph Metrics
for subj = 1:n_subj;

    % grab the test matrix
    test = cmat(:, :, subj);
    dims = size(test);

    % -- connectivity strength / regional connectivity
    %    this is the sum of all correlations
    reg_conn = sum(test) ./ (dims(1)-1);
    % -- regional diversity
    %    this is the sum of correlation(i) - sum of all correlations
    var_conn = sum(test - repmat(reg_conn, [dims(1), 1]).^2) ./ (dims(1)-1);
    % -- global integration:
    %       ratio of 1st eigenvalue to the sum of all other eigenvalues
    [pca_coff, pca_lat] = pcacov(test);
    glo_int = pca_lat(1) / sum(pca_lat(2:end));
    % init output arrays
    n_deg = zeros(14, dims(1));
    n_eff = zeros(14, dims(1));
    n_cls = zeros(14, dims(1));
    n_rnk = zeros(14, dims(1));
    n_par = zeros(14, dims(1));
    n_div = zeros(14, dims(1));

    % not working yet... need to align CI membership...
    n_cid = zeros(14, dims(1));

    g_smw = zeros(14, 1);
    g_eff = zeros(14, 1);
    g_rob = zeros(14, 1);
    g_mod = zeros(14, 1);

    % graphs thresholded at: 37-50% -- average output across 1% increments
    x = 1;
    for t = 37:50;

        % threshold the graph
        tmp_w = threshold_proportional(test, t/100);
        
        % binarize the graph
        tmp_b = tmp_w;
        tmp_b(tmp_b > 0) = 1;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % NODE STATS
        %
        % node degree
        n_deg(x, :) = degrees_und(tmp_b);
        
        % node efficiency
        n_eff(x, :) = efficiency_wei(tmp_w, 1);
        
        % node clustering coefficient
        n_cls(x, :) = clustering_coef_wu(tmp_w);

        % pagerank centrality
        n_rnk(x, :) = pagerank_centrality(tmp_b, 0.85);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % WHOLE-GRAPH STATS
        %
        % graph efficiency
        g_eff(x) = efficiency_wei(tmp_w, 0);

        % graph small-worldness (NB: 1/efficiency is ~ characteristic path len).
        rand_g = randmio_und(tmp_w, 100);
        rand_e = efficiency_wei(rand_g, 0);
        rand_c = mean(clustering_coef_wu(rand_g));
        g_smw(x) = (mean(n_cls(x, :)) / rand_c) / (rand_e / g_eff(x));
        
        % graph robustness
        % CALCULATED AS THE SIZE OF THE LARGEST CONNECTED COMPONENT
        % AFTER REMOVAL OF THE LARGEST DEGREE-NODE k
        % APPROXIMATE INTEGRAL of SIZE / # REMOVED CURVE
        n_nodes = length(find(tmp_b) > 0);
        s_curve = zeros(length(n_nodes), 1);
        tmp_r = tmp_b;
        
        for n = 1:n_nodes;
            
            % find the maximum connected component
            [comps, comp_sizes] = get_components(tmp_r);
            s_curve(n) = max(comp_sizes);
            
            % find the node with maximum degree
            tmp_k = degrees_und(tmp_r);
            idx_k = find(tmp_k == max(tmp_k));
            
            % if we have a tie, use the first one
            if length(idx_k) > 1;
                idx_k = idx_k(1);
            end
        
            % destroy the identisfied node (assumes symmetric)
            tmp_r(idx_k, :) = 0;
            tmp_r(:, idx_k) = 0;

        end
        
        % take integral of the s/n curve to approximate network robustness.
        g_rob(x) = trapz(s_curve);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % CLIQUES
        %
        % graph modularity
        % CALCULATED USING THE LOUVAIN ALGO
        [Ci, Q] = modularity_louvain_und_sign(tmp_w);
        g_mod(x) = Q;

        % participation coefficient
        % measures each node's intermodular communication
        n_par(x, :) = participation_coef(tmp_w, Ci);

        % diversity coefficient
        % entropy-based measure of intramodular communication
        [div_pos, div_neg] = diversity_coef_sign(tmp_w, Ci);
        n_div(x, :) = div_pos; % disregard negative vibes yo

        % iterate the counter
        x = x + 1;
    
    end

    % take mean values across all thresholds
    n_deg = mean(n_deg);
    n_eff = mean(n_eff);
    n_cls = mean(n_cls);
    n_rnk = mean(n_rnk);
    n_par = mean(n_par);
    n_div = mean(n_div);

    g_smw = mean(g_smw);
    g_eff = mean(g_eff);
    g_rob = mean(g_rob);
    g_mod = mean(g_mod);

    % generate output array
    output = [n_deg, n_eff, n_cls, n_rnk, n_par, n_div, ...
              g_smw, g_eff, g_rob, g_mod];

    % append results to the output matrix
    if exist('OUT') == 0;
        OUT = output;
    else;
        OUT = [OUT; output];
    end

end

% get subject IDs
IDs = dlmread([ filetype '_FNC.csv'], ',', [1, 0, -1, 0]);
OUT = [IDs, OUT];

% write the output for all subjects
dlmwrite([ filetype '_FNC_graph.csv'], OUT, 'delimiter', ',', 'precision', 25);