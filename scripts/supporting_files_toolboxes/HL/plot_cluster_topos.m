function plot_cluster_topos(clusterdata,origdata,timepoint,p_plot,p_cdt)
% clusterdata = path to where the clusters that have gone through
% permutation are saved
% origdata = path to the power data from which the clusters were obtaind
% timepoint = which pre-stim timepoint (1 to 4) to plot
% p_plot = the cluster's p-value below which it should be plotted
% p_cdt = the cluster-defining threshold used to compute clusterdata

si.path_base = '/isilon/LFMI/VMdrive/Lua/MEG_alpha_1-f_SCP_PLoSBiol_2024/scripts/supporting_files_toolboxes/HL/';
addpath('/isilon/LFMI/VMdrive/Lua/MEG_alpha_1-f_SCP_PLoSBiol_2024/scripts/supporting_files_toolboxes/HL/');
addpath('/isilon/LFMI/VMdrive/Lua/MEG_alpha_1-f_SCP_HLTP/supporting_files_toolboxes/fieldtrip-20170509');

original_clusters = load(clusterdata);
cluster_data = original_clusters.clusters_orig{timepoint};
sig_clusters = find(cluster_data.cluster_pval < p_plot);

original_data = load(origdata);
original_data = original_data.data;

nSubs = length(original_data)/2;
nSensors = length(original_data{1}(1,:,1));
nTimepoints = length(original_data{1}(1,1,:));

alldat_trialmeans = zeros(nSubs,nTimepoints,2,nSensors); %for computing Wilcoxon stats below: sub x time x cond x sensor
for sub = 1:nSubs
    for time = 1:nTimepoints
       alldat_trialmeans(sub,time,1,:) = nanmean(original_data{(sub-1)*2+1}(:,:,time),1);
       alldat_trialmeans(sub,time,2,:) = nanmean(original_data{(sub-1)*2+2}(:,:,time),1);
    end
end

topo_p_all = zeros(nSensors,nTimepoints);
topo_stat_all = zeros(nSensors,nTimepoints);
for sensor = 1:nSensors
    for time = 1:nTimepoints
        seen = alldat_trialmeans(:,time,1,sensor)';
        unseen = alldat_trialmeans(:,time,2,sensor)';
        stat = wilcoxon(seen(~isnan(seen)),unseen(~isnan(unseen)));
        topo_p_all(sensor,time) = stat.p;
        topo_stat_all(sensor,time) = stat.W;
    end
end

topo_p = topo_p_all(:,timepoint);
topo_stat = topo_stat_all(:,timepoint);

clusters = find_clusters_LK(topo_stat, topo_p, p_cdt, 0, si);
clusters = clusters.topo_cluster;

dat = zeros(nSensors,1);
for i=1:length(sig_clusters)
    clust = sig_clusters(i);
    ind = find(ismember(clusters,clust));
    dat(ind) = clust;
end

topodat = make_ft_struct_HLTP(dat, 'timelock',si);

cfg = [];
cfg.layout    = 'CTF275.lay';
cfg.comment   = 'no';
cfg.colorbar  = 'no';
cfg.style     = 'fill';
cfg.interpolation = 'nearest';

figure;
ft_topoplotER(cfg, topodat);