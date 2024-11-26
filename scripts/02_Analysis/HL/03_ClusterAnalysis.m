si.path_base = '/isilon/LFMI/VMdrive/Lua/MEG_alpha_1-f_SCP_PLoSBiol_2024/scripts/supporting_files_toolboxes/HL/';
addpath('/isilon/LFMI/VMdrive/Lua/MEG_alpha_1-f_SCP_HLTP/supporting_files_toolboxes/fieldtrip-20170509');
data_dir = '/isilon/LFMI/VMdrive/Lua/MEG_alpha_1-f_SCP_PLoSBiol_2024/data/HL/';
addpath('/isilon/LFMI/VMdrive/Lua/MEG_alpha_1-f_SCP_PLoSBiol_2024/scripts/supporting_files_toolboxes/HL/');

alpha_all = load([data_dir,'alpha_per_subject.mat']);
alpha_all = alpha_all.data;
beta_all = load([data_dir,'beta_per_subject.mat']);
beta_all = beta_all.data;

alpha_clusters = runclustering(alpha_all,0.05,[data_dir, 'alpha_clusters.mat']);
beta_clusters = runclustering(beta_all,0.05,[data_dir, 'beta_clusters.mat']);

%% Transform data into readable Python format for subsequent correlation with DVs
alpha_clusters = load('../data/alpha_clusters.mat');
beta_clusters = load('../data/beta_clusters.mat');

alpha_clusters = alpha_clusters.clusters_orig;
alpha_c1 = alpha_clusters{2}.cluster_sensors{1};
alpha_c2 = alpha_clusters{4}.cluster_sensors{1};
alpha_cluster_sensors = {alpha_c1,alpha_c2};

beta_clusters = beta_clusters.clusters_orig;
beta_c1 = beta_clusters{3}.cluster_sensors{1};
beta_c2 = beta_clusters{4}.cluster_sensors{1};
beta_cluster_sensors = {beta_c1,beta_c2};

save([data_dir, 'alpha_cluster_sensors.mat'],'alpha_cluster_sensors');
save([data_dir, 'beta_cluster_sensors.mat'],'beta_cluster_sensors');
