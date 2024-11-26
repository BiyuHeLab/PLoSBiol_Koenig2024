%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%triggers definitions and break a
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%recorded trial continuously 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%into multiple trials based only on the
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%triggers
function [trl,event]=breaktrials_lk(cfg)
%cfg.dataset='XHRCSMPZ_VisThresh_20111025_07.ds', actually the input must
%be of cfg.dataset='ds',
ft_defaults;
hdr   = ft_read_header(cfg.dataset); % this is essentially for the sampling rate 600 only
event = ft_read_event(cfg.dataset);  % this will give the triggers from numberious trigger-channels

% search for "trigger=UPPT001" events. because I use only this one for the
% trigger channel but not others

trigger = [event(strcmp('UPPT001', {event.type})).value]'; %trigger values 128, 112,96
sample=[event(strcmp('UPPT001', {event.type})).sample]';% trigger time 
% determine the number of samples before and after the trigger hdr.Fs=600
% so pretrig is defined in terms of seconds
pretrig  = round(cfg.trialdef.pre  * hdr.Fs); %this convert seconds to index points
posttrig =  round(cfg.trialdef.post * hdr.Fs);

% look for the combination of a trigger "7" followed by a trigger "64" 
% for each trigger except the last one
trl = [];
for j = 1:7:(length(trigger)) %pick up the 128 starting triggers
 %trg1 = trigger(j);

    trlbegin = sample(j) - pretrig;     % the following 3 lines define a trl in sampling points unit 
    trlend   = sample(j) + posttrig;     %property which is requied later ft_defineral   
    offset   = -hdr.Fs*ceil(cfg.trialdef.pre);                       % this says the onset of the stimulus is actually 600 points after the trial begining
    newtrl   = [trlbegin trlend offset];
    trl      = [trl; newtrl];
  end