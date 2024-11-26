function clusters_orig = runclustering(data_all,p_cdt,savename) 
% data_all is a list of length 2 * nSubs, with the ordering of
% (seen,unseen) for each subject, with each list element being a matrix of
% the format trials x sensors x timepoints
% p_cdt is the p-value used as the cluster-defining threshold
% savename is the .mat name to save the clusters to

% Initialize variables
nSubs = length(data_all)/2;
nSensors = length(data_all{1}(1,:,1));
nTimepoints = length(data_all{1}(1,1,:));
nReps = 1000;
plot_option=0; %plotting not working - check why
si.path_base = '/isilon/LFMI/VMdrive/Lua/MEG_alpha_1-f_SCP_PLoSBiol_2024/scripts/supporting_files_toolboxes/HL/';
fileID = fopen([si.path_base, 'CTF275labels.txt']);
label = textscan(fileID,'%s');
label = label{1};
label([33 173 192]) = [];
load([si.path_base, 'ctf275_neighb.mat']);
neighbors = neighbours;
neighbors([33 173 192]) = [];

% Extract data for each subject and timepoint
dv_orig = {};
alldat_trialmeans = zeros(nSubs,nTimepoints,2,nSensors); %for computing Wilcoxon stats below: sub x time x cond x sensor
for sub = 1:nSubs
    for time = 1:nTimepoints
       dv_orig{sub}{time} = horzcat(transpose(data_all{(sub-1)*2+1}(:,:,time)),transpose(data_all{(sub-1)*2+2}(:,:,time)));
       alldat_trialmeans(sub,time,1,:) = nanmean(data_all{(sub-1)*2+1}(:,:,time),1);
       alldat_trialmeans(sub,time,2,:) = nanmean(data_all{(sub-1)*2+2}(:,:,time),1);
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

%% Permutation
for timepoint = 1:nTimepoints; 
    topo_p = topo_p_all(:,timepoint);
    topo_stat = topo_stat_all(:,timepoint);
    
    % find clusters in the original data topo
    clusters_orig{timepoint} = find_clusters_LK(topo_stat, topo_p, p_cdt, plot_option, si, label, neighbors);
    
    disp(['timepoint: ' num2str(timepoint)])
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
                seen_shuffle(i_sub) = nanmean(dv_orig{i_sub}{timepoint}(i_sensor, ind(1:length(data_all{(i_sub-1)*2+1}(:,1,1)))));
                unseen_shuffle(i_sub) = nanmean(dv_orig{i_sub}{timepoint}(i_sensor, ind((length(data_all{(i_sub-1)*2+1}(:,1,1))+1):(length(data_all{(i_sub-1)*2+1}(:,1,1))+length(data_all{(i_sub-1)*2+2}(:,1,1))))));
            end
            
            statshuff = wilcoxon(seen_shuffle(~isnan(seen_shuffle))',unseen_shuffle(~isnan(unseen_shuffle))');
            
            topo_p_shuffle(i_sensor)    = statshuff.p;
            topo_stat_shuffle(i_sensor) = statshuff.W;
        end
        
        % find clusters in the shuffled data topo and get the max cluster stat
        clusters_shuffle{i_rep} = find_clusters_LK(topo_stat_shuffle, topo_p_shuffle, p_cdt, plot_option, si, label, neighbors);
        shuffleMaxStat(i_rep)   = clusters_shuffle{i_rep}.maxStatSumAbs;
        
    end
    
    %% calculate p-values for each cluster in the original data set
    %  on the basis of the estimated null distribution from the shuffled data set
    for i = 1:clusters_orig{timepoint}.nClusters
        pval = sum(abs(shuffleMaxStat) >= abs(clusters_orig{timepoint}.cluster_statSum(i))) / nReps;
        clusters_orig{timepoint}.cluster_pval(i) = pval;
    end
end
%save(['betaclusters_corrected_time_' num2str(timepoint) '.mat'],'clusters_orig');

save(savename,'clusters_orig');