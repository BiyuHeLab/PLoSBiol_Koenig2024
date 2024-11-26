function clusters = find_clusters_LK(topo_stat, topo_p, p_thresh, plot_option, si, label, neighbors)
% clusters = find_clusters(topo_stat, topo_p, p_thresh, plot_option, label, neighbors)
% 
% Find the spatial clusters in a topo which surpass some statistical threshold.
% 
% inputs
% ------
% topo_stat   - array of length nSensors which holds the statistic of interest (e.g. t, F) at each sensor
% topo_p      - array of length nSensors which holds the p-value corresponding to topo_stat
% p_thresh    - p-value below which sensors will be considered for cluster analysis
% plot_option - if 1, plot the final topo of the clusters
%               if 2, plot the cluster topo each time a new sensor is added to a cluster
% label       - cell array of labels for MEG sensors. if not specified, loads default list of 271
% neighbors   - struct of neighbors for each MEG sensor, as output from ft_prepare_neighbours.
%               if not specified, loads neighbors of the default list of 271 sensors as output
%               by ft_prepare_neighbours using the CTF275_neighb.mat template
%             
% output
% ------
% clusters.inputs          - holds the inputs used in the find_cluster function call
% clusters.topo_cluster    - array of length nSensors which holds cluster ID at each sensor 
%                            (0 for sensors not belonging to any cluster)
% clusters.nClusters       - number of clusters found
% clusters.cluster_sensors - cell array of length nClusters where cell i holds the indeces for 
%                            sensors belonging to cluster i
% clusters.cluster_size    - array of length nClusters holding the number of sensors in each cluster
% clusters.cluster_statSum - array of length nClusters holding the sum of topo_stat for each cluster 
% clusters.maxSize         - maximum cluster size
% clusters.maxStatSumPos   - maximum statSum among all clusters with a positive statSum
% clusters.maxStatSumNeg   - minimum statSum among all clusters with a negative statSum
% clusters.maxStatSumAbs   - maximum absolute value of statSum

%run '/isilon/LFMI/archive/Projects/2017_PCB_Gabor_Trajectory_Brian/TRA_set_path.m'

%% define input defaults

if ~exist('plot_option', 'var') || isempty(plot_option)
    plot_option = 0;
end

if ~exist('label', 'var') || isempty(label)
    fileID = fopen('/isilon/LFMI/VMdrive/Lua/MEG_alpha_1-f_SCP_HLTP/supporting_files_toolboxes/CTF275labels.txt');
    label = textscan(fileID,'%s');
    label = label{1};
    label([33 173 192]) = [];
end

if ~exist('neighbors', 'var') || isempty(neighbors)
     load('/isilon/LFMI/VMdrive/Lua/MEG_alpha_1-f_SCP_HLTP/supporting_files_toolboxes/ctf275_neighb.mat');
     neighbors = neighbours;
     neighbors([33 173 192]) = [];
end

%% find clusters

% find sensors whose statistic passes the threshold for cluster inclusion
topo_thresh = topo_p < p_thresh;
thresh_ind  = find(topo_thresh==1);

% initialize a topo which will hold cluster information
topo_cluster = zeros(size(topo_p));

clusterID = 0;
for i = 1:length(thresh_ind)
    
    ind = thresh_ind(i);
    
    if topo_cluster(ind) == 0
        clusterID    = clusterID + 1;
        cluster_sign = sign(topo_stat(ind));
        
        topo_cluster = search_neighbors_LK(ind, clusterID, cluster_sign, topo_stat, topo_thresh, topo_cluster, plot_option, label, neighbors);
    end
end

%% process and package output

% store inputs
clusters.inputs.topo_stat    = topo_stat;
clusters.inputs.topo_p       = topo_p;
clusters.inputs.p_thresh     = p_thresh;
clusters.inputs.label        = label;
clusters.inputs.neighbors    = neighbors;

% save cluster topo
clusters.topo_cluster = topo_cluster;


% save stats for each cluster
nClusters          = clusterID;
clusters.nClusters = nClusters;

maxSize       = -Inf;
maxStatSumPos = -Inf;
maxStatSumNeg =  Inf;

for i = 1:nClusters
    
%     clusters.clusterStats(i).clusterID = i;
    
    % get indeces to sensors for the current cluster
    ind = find(topo_cluster == i);
    clusters.cluster_sensors{i} = ind;
    
    % get size of current cluster
%     clusters.clusterStats(i).clusterSize = length(ind);
    clusters.cluster_size(i) = length(ind);
    
    % get sum of topo_stat for current cluster
%     clusters.clusterStats(i).statSum = sum(topo_stat(ind));
    clusters.cluster_statSum(i) = sum(topo_stat(ind));

    
    % update max size
    if clusters.cluster_size(i) > maxSize
        maxSize = clusters.cluster_size(i);
    end
    
    
    % update max stat sum
    if clusters.cluster_statSum(i) >= 0       
        if clusters.cluster_statSum(i) > maxStatSumPos
            maxStatSumPos = clusters.cluster_statSum(i);
        end
    
    else
        if clusters.cluster_statSum(i) < maxStatSumNeg
            maxStatSumNeg = clusters.cluster_statSum(i);
        end
    end
    
end

% save info on maximum cluster stats

if maxStatSumPos == -Inf && maxStatSumNeg == Inf
    maxSize       = 0;
    maxStatSumPos = 0;
    maxStatSumNeg = 0;
    maxStatSumAbs = 0;

elseif maxStatSumPos == -Inf
    maxStatSumPos = 0;
    maxStatSumAbs = maxStatSumNeg;
    
elseif maxStatSumNeg == Inf
    maxStatSumNeg = 0;
    maxStatSumAbs = maxStatSumPos;
    
else
    if maxStatSumPos < abs(maxStatSumNeg);
        maxStatSumAbs = maxStatSumNeg;
    else
        maxStatSumAbs = maxStatSumPos;
    end
end

clusters.maxSize       = maxSize;
clusters.maxStatSumPos = maxStatSumPos;
clusters.maxStatSumNeg = maxStatSumNeg;
clusters.maxStatSumAbs = maxStatSumAbs;



%% plot the clusters

if plot_option == 1
    datfortopo=topo_cluster;
    
    dat = make_ft_struct_HLTP(datfortopo, 'timelock',si);

    cfg = [];
    cfg.layout    = 'CTF275.lay';
    cfg.comment   = 'no';    
    cfg.colorbar  = 'yes';
    cfg.style     = 'fill';
    cfg.interpolation = 'nearest';

    figure; 
    ft_topoplotER(cfg, dat);
end

end
