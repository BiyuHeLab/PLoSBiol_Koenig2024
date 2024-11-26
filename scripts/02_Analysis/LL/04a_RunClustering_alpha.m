%% Updating script to reflect how pvalues are calculated for each cluster and which alpha power is loaded in (no AUC power value if no alpha peak is found ('allpowerAUC_allsensorsWAVvsFFT.mat' had AUC values even if there was no peak))

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
alpha_all = load([data_dir, 'allpostfooofpower.mat']);
alpha_all = alpha_all.Posc_all_dict; %This has dimensions 6 (alpha power, 1/f power in alpha band, total power in alpha band, beta power, 1/f power in beta band, total power in beta band) x nTrials x nTimepoints x nSensors

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

for timepoint = 1:4; 
    topo_p = topo_p_all(:,timepoint);
    topo_stat = topo_stat_all(:,timepoint);
    p_thresh = 0.05;
    
    % find clusters in the original data topo
    plot_option=0; %plotting not working - check why
    clusters_orig{timepoint} = find_clusters_LK(topo_stat, topo_p, p_thresh, plot_option, si, label, neighbors);
    
    %% analysis of repeatedly shuffled data set
    for i_rep = 1:nReps
         if mod(i_rep,100) == 0
            disp(['rep#: ' num2str(i_rep) '/' num2str(nReps)])
         end
        % construct shuffle permutation for current iteration
        shuffle_ind = [];
        for i_sub = 1:nSubs
            nTrials = length(dv_orig{i_sub}{1}(1,:));
            shuffle_ind{i_sub} = randperm(nTrials);
        end
        
        for i_sensor = 1:nSensors
            seen_shuffle = zeros(nSubs,1);
            unseen_shuffle = zeros(nSubs,1);
            % apply the shuffling to the original data
            for i_sub = 1:nSubs
                ind = shuffle_ind{i_sub};
                seen_shuffle(i_sub) = nanmean(dv_orig{i_sub}{timepoint}(i_sensor, ind(1:cond_n{i_sub}(1))));
                unseen_shuffle(i_sub) = nanmean(dv_orig{i_sub}{timepoint}(i_sensor, ind((cond_n{i_sub}(1)+1):(cond_n{i_sub}(1)+cond_n{i_sub}(2)))));
            end
            
            statshuff = wilcoxon(seen_shuffle',unseen_shuffle');
            
            topo_p_shuffle(i_sensor)    = statshuff.p;
            topo_stat_shuffle(i_sensor) = statshuff.W;
        end
        
        % find clusters in the shuffled data topo and get the max cluster stat
        clusters_shuffle{i_rep} = find_clusters_LK(topo_stat_shuffle, topo_p_shuffle, p_thresh, plot_option, si, label, neighbors);
        shuffleMaxStat(i_rep)   = clusters_shuffle{i_rep}.maxStatSumAbs;
        
    end
    
    %% calculate p-values for each cluster in the original data set
    %  on the basis of the estimated null distribution from the shuffled data set
    for i = 1:clusters_orig{timepoint}.nClusters
        pval = sum(abs(shuffleMaxStat) >= abs(clusters_orig{timepoint}.cluster_statSum(i))) / nReps;
        clusters_orig{timepoint}.cluster_pval(i) = pval;
    end
end
save([data_dir, 'alphaclusters.mat'],'clusters_orig');
