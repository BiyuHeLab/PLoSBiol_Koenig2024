si.path_base = '/isilon/LFMI/VMdrive/Lua/MEG_alpha_1-f_SCP_PLoSBiol_2024/scripts/supporting_files_toolboxes/HL/';
addpath('/isilon/LFMI/VMdrive/Lua/MEG_alpha_1-f_SCP_HLTP/supporting_files_toolboxes/fieldtrip-20170509');
data_dir = '/isilon/LFMI/VMdrive/Lua/MEG_alpha_1-f_SCP_PLoSBiol_2024/data/HL/';
addpath('/isilon/LFMI/VMdrive/Lua/MEG_alpha_1-f_SCP_PLoSBiol_2024/scripts/supporting_files_toolboxes/HL/');

plot_cluster_topos([data_dir,'alpha_clusters.mat'],[data_dir,'alpha_per_subject.mat'],2,0.1,0.05)
plot_cluster_topos([data_dir,'alpha_clusters.mat'],[data_dir,'beta_per_subject.mat'],4,0.1,0.05)

plot_cluster_topos([data_dir,'beta_clusters.mat'],[data_dir,'beta_per_subject.mat'],3,0.1,0.05)
plot_cluster_topos([data_dir,'beta_clusters.mat'],[data_dir,'beta_per_subject.mat'],4,0.1,0.05)