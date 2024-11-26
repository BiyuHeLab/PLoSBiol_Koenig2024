close all
clear all

%Import information about subjects
run '/isilon/LFMI/VMdrive/Lua/MEG_alpha_1-f_SCP_PLoSBiol_2024/scripts/supporting_files_toolboxes/LL/TRA_set_path.m'
si = TRA_subject_info_with_ICA([], path_base);
data_path = '/isilon/LFMI/archive/gago/disk2/Qili/';
path_out = '/isilon/LFMI/VMdrive/Lua/MEG_alpha_1-f_SCP_PLoSBiol_2024/data/LL/Preprocessed/';
subs = si.sub_list;
addpath('/isilon/LFMI/VMdrive/Lua/MEG_alpha_1-f_SCP_PLoSBiol_2024/scripts/supporting_files_toolboxes/LL/fieldtrip-20170509');
addpath('/isilon/LFMI/VMdrive/Lua/MEG_alpha_1-f_SCP_PLoSBiol_2024/scripts/supporting_files_toolboxes/LL/');

for sub = 1:length(subs)
    ft_defaults
    eprimedirectory={[data_path subs{sub} '/eprime']};
    datadirectory={[data_path subs{sub} '/processed']};
    addpath(eprimedirectory{1})
    addpath(datadirectory{1})
    addpath([data_path subs{sub} '/processed/figures'])
    opengl software
    pointer=1;
    %ini_mag_eprime
    run([data_path subs{sub} '/subobj_anova.m']);
    %%%%%%%%%%%%%%%%%%%%%%%%%% input session nos
    for megpost=megtest; %meg file postfix number
        SCP = 1; %If preprocessing for SCP set this to 1, otherwise if processing for full band, run with SCP = 0
        %%%%%%%%%%%input the raw meg fiel
        megfile=[data_path subs{sub} '/' strcat([namein1,'_0',num2str(megpost),'.ds'])];
        %e_prime_file='gabors-0126-fixed-duration-1-';
        eprimefile=[data_path subs{sub} '/' strcat([e_prime_file,num2str(eprimeset(pointer)),'.xls'])];
        onetrialintotrials(megfile,[0.05 300],subs{sub},megpost,SCP);
        pointer=pointer+1; %eprime xls post fix number
    end
end
