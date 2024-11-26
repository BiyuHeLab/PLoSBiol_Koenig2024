function topo_cluster = search_neighbors_LK(ind, clusterID, cluster_sign, topo_stat, topo_thresh, topo_cluster, plot_option, label, neighbors)
%%

% if current sensor passes the statistical threshold,
% AND it's the same sign as the current cluster,
% AND it hasn't been marked yet...
if topo_thresh(ind) == 1 && sign(topo_stat(ind)) == cluster_sign && topo_cluster(ind) == 0
    
    % mark current sensor as belonging to current cluster
    topo_cluster(ind) = clusterID;
    
    if plot_option == 2
        dat = make_ft_struct(topo_cluster, 'timelock');
        
        cfg = [];
        cfg.layout    = 'CTF275.lay';
        cfg.comment   = 'no';
        cfg.colorbar  = 'yes';
        cfg.style     = 'fill';
        cfg.interpolation = 'nearest';

        figure(100); 
        ft_topoplotER(cfg, dat);
    end

    
    % search the neighbors of the current sensor
    neighb = neighbors(ind).neighblabel;
    for i = 1:length(neighb)
        neighb_ind = find(strcmp(neighb{i}, label));
        
        % neighb_ind might be empty if a neighbor of the current sensor in
        % the full CTF 275 sensor layout is listed but not included in the
        % 271 sensor set we have been working with
        if ~isempty(neighb_ind)
            topo_cluster = search_neighbors_LK(neighb_ind, clusterID, cluster_sign, topo_stat, topo_thresh, topo_cluster, plot_option, label, neighbors);
        end        
    end
    
end


end