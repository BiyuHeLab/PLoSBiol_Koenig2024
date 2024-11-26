% must be used in the meg foldder, must have a processed folder, output is
% the separated trials. input is the name of the raw meg data.answerstruct
% has two fields answerstruct.subja (value yes or no) or trlp(value:trials
% or catchtrials. interesting trials is from eprime_trials_group function
% which seperated the yes trials and no trials no as interesting trials
function onetrialintotrials(s,band,subject,megpost,SCP)
path_out = ['/isilon/LFMI/VMdrive/Lua/MEG_alpha_1-f_SCP_PLoSBiol_2024/data/LL/Preprocessed/', subject];

% s='XHRCSMPZ_VisThresh_20111025_07.ds';

%%%%set default value of the interestingtrials
% i=1; 
% while i<=length(interestingtrials)
%   interestingtrials = 0; 
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%interestingtrials is only an array of trial nos
% if isempty(interestingtrials)
%     fprintf('%s\n', 'no trials in this categroy!!!! try other category in slowpotential_topoplot')
%     pause
% else
b_and=[]; %default is no bandpass only low and high pass.
cfg=[];
cfg.dataset=s;
event = ft_read_event(cfg.dataset); 
trigger = [event(strcmp('UPPT001', {event.type})).value]'; %trigger values 128, 
sample=[event(strcmp('UPPT001', {event.type})).sample]';

cfg.trl=[1 sample(end) 0];
cfg.channel={'MEG','-MRF43','-MRO13'};     %exclude broken channels
cfg.bsfilter='yes';
cfg.bsfreq=[59.5 60.5]; %suggested by fieldtrip for larger dataset emails forum from beth
% cfg.dftfilter = 'yes';
% cfg.dftfreq = [60 120];

trials=ft_preprocessing(cfg);

cfg=[];
cfg.bsfilter='yes';
cfg.bsfreq=[119.5 120.5]; %remove power harmonics
trials=ft_preprocessing(cfg,trials);

 
cfg=[];
cfg.demean='yes'; %zeromean and detrending
cfg.detrend='yes';
%cfg.hpfilttype='but';

if SCP
    cfg.bpfilter='yes';
    cfg.bpfreq=band;
    %cfg.bpfilttype='biyu';
    cfg.bpfiltord = 5;
    b_and='band'; %without powerline removal date before 01/2013
    b_and='band_powline'; %after 02/08/2013
else 
    cfg.lpfilter='yes';
    cfg.lpfreq=35;  %low frequency filter including theta band 35; %%11/1/21 this is wrong: should be 150 for full band. Corrected when it was run.
    cfg.lpfiltord=4;
    cfg.hpfilter='yes';
    cfg.hpfreq=0.05;%high pass filter used to be 0.05 set 1 is better to increase snr
    cfg.hpfiltord=4;
    %fir is very slow, but is unstable for small values
end

cfg.trl=[1 sample(end) 0];
%cfg.channel={'MEG','-MRF43','-MRO13'};     %exclude broken channels
triggered_trials=ft_preprocessing(cfg,trials);

%save(strcat([s(1:end-3) 'lpass' b_and]),'triggered_trials');


%the above is the truncation of the test part and apply filters, the
%following is for epoch
cfg=[];
cfg.dataset=s; %raw data 

cfg.trialdef.pre=1.85; %before 128trigger 1s fttrip define the trigger to be onset%used to be 0.85
cfg.trialdef.post=3.15;%after  128 trigger 3 seconds

[trial event]=breaktrials_lk(cfg); %define the cfg.trl property to separate the trials
cfg=[];

cfg.trl=trial;

triggered_trials=ft_redefinetrial(cfg,triggered_trials);

name=[path_out '2sec_separated' b_and '_' num2str(megpost) '.mat'];
save(name, 'triggered_trials')  %axial gradient all separated trials saved

end