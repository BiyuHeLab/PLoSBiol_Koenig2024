%% initialize paths for fieldtrip and scripts

restoredefaultpath;

%%% set this to the path of the main project folder 
path_base = textread('path_base.txt','%s');
path_base = path_base{1};

addpath( path_base );
addpath( genpath([path_base 'analysis/']) );
addpath( genpath([path_base 'scripts/']) );

ft_defaults