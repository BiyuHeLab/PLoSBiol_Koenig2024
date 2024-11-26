function data_struct = make_ft_struct_HLTP(data, data_type, si)
% data_struct = make_ft_struct(data, data_type)
% 
% Make a data struct in Fieldtrip format e.g. for plotting a topo.
% data = MEG data. should be of size nSensors x nSamples
% data_type = type of Fieldtrip data struct. can be 'timelock' or 'freq'

load([si.path_base 'label272.mat']);
load([si.path_base 'grad.mat']);

nSamples = size(data,2);

switch data_type
    case 'timelock'
        
        data_struct.dimord = 'chan_time';
        data_struct.label  = label272;
        data_struct.grad   = grad;
        data_struct.time   = 0:nSamples-1;
        data_struct.avg    = data;
        
    case 'freq'
        
        data_struct.dimord    = 'chan_freq';
        data_struct.label     = label272;
        data_struct.grad      = grad;
        data_struct.freq      = 0:nSamples-1;
        data_struct.powspctrm = data;

end