%% 042121 Width of wavelet cycles changes across frequencies

clear

addpath('/isilon/LFMI/VMdrive/Lua/MEG_alpha_1-f_SCP_PLoSBiol_2024/scripts/supporting_files_toolboxes/LL/fieldtrip-20170509');

opengl software

fileID = fopen('/isilon/LFMI/VMdrive/Lua/MEG_alpha_1-f_SCP_PLoSBiol_2024/scripts/supporting_files_toolboxes/LL/CTF275labels.txt');
L = textscan(fileID,'%s');
L = L{1};
L([173 192]) = [];

load('/isilon/LFMI/VMdrive/Lua/MEG_alpha_1-f_SCP_PLoSBiol_2024/data/LL/Preprocessed/MEG_by_cond_full_band_prestim_2sec.mat','CONDITION1');
load('/isilon/LFMI/VMdrive/Lua/MEG_alpha_1-f_SCP_PLoSBiol_2024/data/LL/Preprocessed/MEG_by_cond_full_band_prestim_2sec.mat','CONDITION2');

conds = {'seen','unseen'};
nSamples = length(CONDITION1.sub1.ts(1,:,1));
nsubs = 11;

for sub = 1:nsubs
    for cond = 1:2
        sub
        cond
        
        ft_defaults
        
        data = [];
        data.label = L;
        data.fsample = 600;
        data.trial = transpose(squeeze(num2cell(eval(['CONDITION' num2str(cond) '.sub' num2str(sub) '.ts']),[1 2])));
        data.trial = data.trial * 1e15; %convert to fT
        data.time = cell(size(data.trial));
        data.time(:) = {1/600*(-1200:1800)}; %in seconds

        cfg = [];
        cfg.channel    = 'MEG';	                
        cfg.method     = 'wavelet';                
        cfg.width      = linspace(3, 9, length(0:0.8:40)); 
        cfg.output     = 'pow';	
        cfg.keeptrials = 'yes';
        cfg.foi        = 0:0.8:40; % frequency resolution of 0.8 Hz
        cfg.toi        = -2:0.1:3; % temporal resolution of 100 ms      

        TFR = ft_freqanalysis(cfg, data);
        save(['/isilon/LFMI/VMdrive/Lua/MEG_alpha_1-f_SCP_PLoSBiol_2024/data/LL/Preprocessed/power_' num2str(sub) '_' char(conds(cond)) '.mat'],'TFR','-v7.3');
        clear cfg
        clear TFR
    end
end
