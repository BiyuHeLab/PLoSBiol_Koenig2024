%this function epochs Qi's final preprocessed data according to a list of
%conditions, averages timeseries within the epochs (mean filter), and
%returns the filtered epochs with a table showing which trials were kept

%inputs
%subject = subject number
%catseq = list of conditions to keep
%window = temporal window within which to average time series.  if
%window==1 then no averaging is done
function [TS1m,fintable] = extract_epochs_2sec(subject,catseq,window,band_name,path_base)

%%% Brian new code for isilon file locations

% % % filebase = '/data/disk2/Qili/';
% % filebase = '/data/gagodisk2/Qili/';
% % addpath(genpath([filebase 'processing/']))
% % 
% % %decipher Qi's session coding and file names
% % eval(['cd ' filebase subject ';']);
% % subobj_anova
% % 
% % %%%%%%%%%change 04/19/2015 for filtering
% % %megpost = '_separatedband_powline'; 
% % %%%%%%%
% % 
% % % cd /data/gogohome/bariaat/QiMegTest/AlexMatlabScripts/
% % cd /home/bariaat/QiMegTest/AlexMatlabScripts/

% cd('/isilon/LFMI/archive/Projects/2017_Gabor_Trajectory_Brian/');
% brian_ft_path

% addpath(genpath('/isilon/LFMI/archive/Projects/2017_Gabor_Trajectory_Brian/qi_processing/'));

% si = subject_info_trajectory(subject);
si = TRA_subject_info_with_ICA(subject, path_base);

% cd('/isilon/LFMI/archive/Projects/2017_Gabor_Trajectory_Brian/data/');

%load index of good trials
% % % file = [filebase subject '/arti.mat'];
file = [si.path_data_sub 'arti.mat'];

run([si.path_data_sub 'subobj_anova.m']);

eval(['load ' file]);


if any(strcmp(si.sub_list, subject))
    numsens = 273;
else
    numsens = 271;
end
%%% end Brian edit

if window==1
    after = 1:3001; %changed from 1:2401 on 3/31/21 by LK - intending to obtain -2 to + 3 sec around stim presentation 
else
    overlap = round(window/2); 
    after = 1:overlap:2401; %temporal chunks of data to average
end 

%outputs
TS1m = []; %3D array of temporally averaged epochs (sensor x time x trial)
fintable = []; %final table of trials of interest for set of epochs
% keyboard
for j = 1:length(eprimeset)

    %find trials of interest
% % %     eprimexls = [filebase subject '/eprime/' e_prime_file int2str(eprimeset(j)) '.xls'];

    eprimexls = [si.path_data_sub 'eprime/' e_prime_file int2str(eprimeset(j)) '.xls'];
    
    preselect = ar{j}; %take only data with minimal artifact
    
    try
        newtable = eprime_sorting(eprimexls,[],preselect);
    catch
        keyboard
    end
    
    
    
    %%%% START EDIT FOR LOADING IN NEW 150 Hz LP ICA DATA
% % %     %load processed MEG data
% % %     M = load([filebase subject '/processed/' namein1 '_0' int2str(megtest(j)) ...
% % %         megpost '.mat']);

    %load processed MEG data
    M = load(['/isilon/LFMI/VMdrive/Lua/MEG_alpha_1-f_SCP_PLoSBiol_2024/data/LL/Preprocessed/' subject '/data_clean_segmented_0' int2str(megtest(j)) '_2secprestim.mat']);
    
    %%%%% END EDIT
    
    
    

    %%%%% changed 4/19/2015 to input powline filtered data
%     newfield = 'data_c';
%     oldfield = 'triggered_trials';
%     [M.(newfield)] = M.(oldfield);
    %%%%%%%
    
    %epoch data according to interested trials
    interesting_trials = newtable_sort(newtable,catseq);
    interesting_trials_mref = cell2mat(newtable(interesting_trials,5));   %THIS IS CORRECTED!! These are the REAL interesting trial numbers! 
    
    %output the table with the master trial number as well
    fintable = [fintable; newtable(interesting_trials,:)]; %reapplying "interesting_trials" to the "newtable" also gives the correct reference to the correct trials

    %initialize new time-series data matrices for temporally averaged
    %signals
    if window ~=1
        TS1 = zeros(size(M.data_c.trial{1},1),length(after)-2, length(interesting_trials));
    else
        TS1 = zeros(size(M.data_c.trial{1},1),length(after), length(interesting_trials));
    end 
    
    
    for k = 1:length(interesting_trials)

        ts = M.data_c.trial{interesting_trials_mref(k)}; %extract the REAL interesting trial
        if window ~=1
            tsm = zeros(numsens,length(after)-2);
            for kk = 1:length(after)-2
                pta = after(kk); ptb = after(kk+2);
                tsm(:,kk) = mean(ts(:,pta:ptb),2); %temporal average of each time series
            end
        else
            tsm = ts;
        end
        TS1(:,1:size(tsm,2),k) = tsm;

    end
    TS1m = cat(3,TS1m,TS1);  
end
