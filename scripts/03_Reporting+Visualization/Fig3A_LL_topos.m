clear

run '/isilon/LFMI/VMdrive/Lua/MEG_alpha_1-f_SCP_PLoSBiol_2024/scripts/supporting_files_toolboxes/LL/TRA_set_path.m'
data_dir = '/isilon/LFMI/VMdrive/Lua/MEG_alpha_1-f_SCP_PLoSBiol_2024/data/LL/';
addpath('/isilon/LFMI/VMdrive/Lua/MEG_alpha_1-f_SCP_PLoSBiol_2024/scripts/supporting_files_toolboxes/LL/');

% subject data
si = TRA_subject_info([], path_base);
nSubs = length(si.sub_list);
nSensors = 273;
fileID = fopen('/isilon/LFMI/VMdrive/Lua/MEG_alpha_1-f_SCP_PLoSBiol_2024/scripts/supporting_files_toolboxes/LL/CTF275labels.txt');
label = textscan(fileID,'%s');
label = label{1};
label([173 192]) = [];
load('/isilon/LFMI/VMdrive/Lua/MEG_alpha_1-f_SCP_PLoSBiol_2024/scripts/supporting_files_toolboxes/LL/ctf275_neighb.mat');
neighbors = neighbours;
neighbors([173 192]) = [];

%load in data
cond_n = load([data_dir, 'trial_numbers_2conds.mat']);
cond_n = cond_n.n;
nReps = 1000;
beta_all = load([data_dir, 'allpostfooofpower.mat']);
beta_all = beta_all.Posc_all_dict; %This has dimensions 6 (alpha power, 1/f power in alpha band, total power in alpha band, beta power, 1/f power in beta band, total power in beta band) x nTrials x nTimepoints x nSensors

dv_orig = {};
alldat_trialmeans = zeros(nSubs,4,2,nSensors); %for computing Wilcoxon stats below: sub x time x cond x sensor
for sub = 1:nSubs
    for time = 1:4
       dv_orig{sub}{time} = transpose(squeeze(horzcat(beta_all{(sub-1)*2+1}(4,:,time,:),beta_all{(sub-1)*2+2}(4,:,time,:))));
       alldat_trialmeans(sub,time,1,:) = nanmean(squeeze(beta_all{(sub-1)*2+1}(4,:,time,:)),1);
       alldat_trialmeans(sub,time,2,:) = nanmean(squeeze(beta_all{(sub-1)*2+2}(4,:,time,:)),1);
    end
end

topo_p_all = zeros(273,4);
topo_stat_all = zeros(273,4);
for sensor = 1:273
    for time = 1:4
        stat = wilcoxon(alldat_trialmeans(:,time,1,sensor)',alldat_trialmeans(:,time,2,sensor)');
        topo_p_all(sensor,time) = stat.p;
        topo_stat_all(sensor,time) = stat.W;
    end
end

%% Plot two beta clusters
clusters_corr = load([data_dir, 'betaclusters.mat']);
timepoint = 1;
clusters_corr = clusters_corr.clusters_orig{timepoint};
sig_clusters = find(clusters_corr.cluster_pval<0.05);
topo_p = topo_p_all(:,timepoint);
topo_stat = topo_stat_all(:,timepoint);
p_thresh=0.05;
clusters = find_clusters_LK(topo_stat, topo_p, p_thresh, 0, si);
clusters = clusters.topo_cluster;

% Cluster 1
dat = zeros(nSensors,1);
i=1;
clust = sig_clusters(i);
ind = find(ismember(clusters,clust));
dat(ind) = clust;

topodat = make_ft_struct(dat, 'timelock',si);

cfg = [];
cfg.layout    = 'CTF275.lay';
cfg.comment   = 'no';
cfg.colorbar  = 'no';
cfg.style     = 'fill';
cfg.interpolation = 'nearest';

figure;
ft_topoplotER(cfg, topodat);

% Cluster 2
dat = zeros(nSensors,1);
i=2;
clust = sig_clusters(i);
ind = find(ismember(clusters,clust));
dat(ind) = clust;

topodat = make_ft_struct(dat, 'timelock',si);

cfg = [];
cfg.layout    = 'CTF275.lay';
cfg.comment   = 'no';
cfg.colorbar  = 'no';
cfg.style     = 'fill';
cfg.interpolation = 'nearest';

figure;
ft_topoplotER(cfg, topodat);