%organizes Qi's preprocessed data into struct files which are used for
%all later analyses

clear
clc

run '/isilon/LFMI/VMdrive/Lua/MEG_alpha_1-f_SCP_PLoSBiol_2024/scripts/supporting_files_toolboxes/LL/TRA_set_path.m'
addpath('/isilon/LFMI/VMdrive/Lua/MEG_alpha_1-f_SCP_PLoSBiol_2024/scripts/supporting_files_toolboxes/LL/');

si = TRA_subject_info_with_ICA([], path_base);

%% organizer settings

% 'main' for main n=11 group
% 'extra' for extra n=6 subjects
sub_group = 'main';

% set band name to 'full_band' for full band analysis or 'band=XX-YY' for
% bandpass analysis b/t XX and YY Hz
band_names = {'full_band'};
%band_names = {'band=0.05-5'}; %% Change this to 

% tinc is the size (in samples) of the moving average window
% - when using a bandpass filter besides full_band, set tinc=1
% - when using a moving average window, set band_name = 'full_band'
% - since sampling frequency is 600 Hz, the size of the moving average window
% in ms is 1000*tinc/600
tinc = 1;
tinc_inms = 1000*tinc/600;

path_out = '/isilon/LFMI/VMdrive/Lua/MEG_alpha_1-f_SCP_PLoSBiol_2024/data/LL/Preprocessed/';


%%

if strcmp(sub_group, 'main')
    subs = si.sub_list;
    suffix = '';
elseif strcmp(sub_group, 'extra')
    subs = si.sub_list_extra;
    suffix = '_extra_subs';
end

% cd('/isilon/LFMI/archive/Projects/2017_Gabor_Trajectory_Brian/data');
% 
% fileID = fopen('subject_list_extra.txt');
% A = textscan(fileID,'%s');
% A = A{1};

% fs = 600; %sample rate



i_b = 1;

band_name = band_names{i_b}

if tinc > 1 && ~strcmp(band_name, 'full_band')
    disp('when tinc > 1 bandpass must be set to full_band. exiting')
    return
end

%list conditions
c1 = {'keeptrial','keepyes'};  % trials_seen
c2 = {'keeptrial','keepno'};  % trials_unseen
Conditions = [c1' c2'];

%struct arrays for holding time series of epoched data for each subject
for i = 1:size(Conditions,2)
    eval(['CONDITION' int2str(i) '=[];']);
end

for i = 1:length(subs)
    tic
    for j = 1:size(Conditions,2)
        [TS,table] = extract_epochs_2sec(subs{i},Conditions(:,j),tinc,band_name,path_base); %filters time series with average window
        eval(['CONDITION' int2str(j) '.sub' int2str(i) '.ts=TS;']);
        eval(['CONDITION' int2str(j) '.sub' int2str(i) '.table=table;']);
    end
    toc
end
% save(['MEGdata_by_cond_tinc' int2str(tinc) '_biyu_150hz_lp.mat'],'CONDITION1','CONDITION2','-v7.3');

if tinc==1
    filename = ['MEG_by_cond_' band_name suffix '_prestim_2sec.mat'];
else
    filename = ['MEG_by_cond_' num2str(tinc_inms) 'ms_mov_avg_win' suffix '.mat'];
end

save([path_out filename],'CONDITION1','CONDITION2','-v7.3');

     
