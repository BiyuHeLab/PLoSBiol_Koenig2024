function TRA_decode_lk(i_band, i_decoding, compute_envelope, extra_subs, extended_baseline)
% TRA_decode(i_band, i_decoding, compute_envelope [, extra_subs] [, extended_baseline])
%
% Decode experimental conditions from MEG data.
%
% INPUTS
% ------
% * i_band determines which frequency band to perform decoding on.
%   i_band = 1 --> 0.05 - 5 Hz
%   i_band = 2 --> 5 - 15 Hz
%   i_band = 3 --> 15 - 30 Hz
%   i_band = 4 --> 0.05 - 30 Hz
%   i_band = 5 --> 30 - 60 Hz
%   i_band = 6 --> 60 - 150 Hz
%
% * i_decoding determines which experimental conditions to decode.
%   i_decoding = 1  --> seen vs unseen
%   i_decoding = 2  --> seen vs unseen for correct trials only
%   i_decoding = 3  --> stimulus left vs right
%   i_decoding = 4  --> stimulus left vs right for seen trials only
%   i_decoding = 5  --> stimulus left vs right for unseen trials only
%   i_decoding = 6  --> response left vs right
%   i_decoding = 7  --> response left vs right for unseen trials only
%   i_decoding = 8  --> correct vs incorrect
%   i_decoding = 9  --> correct vs incorrect for unseen trials only
%   i_decoding = 10 --> seen CORRECT vs unseen INCORRECT
%
% * compute_envelope = 1 --> perform decoding with amplitude envelope of MEG data.
%   compute_enevlope = 0 --> perform decoding with raw MEG timecourse.
%
% * extra_subs is an optional input. If set to 1, the analysis will be
%   conducted on the extra subject data set rather than the main subject
%   data set.
%
% * extended_baseline is an optional input. If set to 1, the analysis will be
%   conducted on MEG data with up to 2 s before stimulus onset rather than
%   the default 1 s.
%
% OUTPUTS
% -------
% 
% Output is saved to {path_base}/analysis/MEG/decoding/results/{band_name}
%
% Computed variables are
% * timevec : a vector of time relative to stimulus onset at each sample.
%
% * svm_acc_time : an n_subs x n_samples matrix of mean decoding accuracy
%   across folds and repetitions.
%
% * ap_sub_time : an n_sensors x n_samples x n_subs matrix of activation 
%   patterns used for decoding at each sample and sensor, averaged across
%   folds and repetitions.
%
% * w_sub_time and b_sub_time : n_sensors x n_samples x n_subs matrices 
%   containing the SVM parameters w and b, averaged across folds and
%   repetitions.


data_dir = '/isilon/LFMI/VMdrive/Lua/MEG_alpha_1-f_SCP_PLoSBiol_2024/data/LL/';

%% analysis parameters

band_names = {'band=0.05-5', 'band=5-15', 'band=15-30', 'band=0.05-30', 'band=30-60', 'band=60-150'};

% decoding options
n_folds = 5; % number of decoding folds
n_reps = 10; % number of repetitions of randomly selected balanced training set
libsvm_options_test  = '-q'; % options for libsvm test set
libsvm_options_train = ['-q -t 0 -c ' num2str(2^-6)]; % options for libsvm training set
i_band=1;
i_decoding=1;
compute_envelope = 0;



%% initialize

% paths
run '/isilon/LFMI/VMdrive/Lua/MEG_alpha_1-f_SCP_PLoSBiol_2024/scripts/supporting_files_toolboxes/LL/TRA_set_path.m'

if ~exist('extra_subs','var')
    extra_subs = 0;
end

if ~exist('extended_baseline','var')
    extended_baseline = 0;
end

% subject data
si = TRA_subject_info([], path_base);

if extra_subs
    n_subs = length(si.sub_list_extra);
else
    n_subs = length(si.sub_list);
end

%% load data

band_name = band_names{i_band};

load([data_dir, 'Preprocessed/MEG_by_cond_band=0.05-5_prestim_2sec.mat']);

% CONDITION 1 --> all seen including catch trials (hits and FAs)
% CONDITION 2 --> all unseen including catch trials (misses and CRs)

%% ampltidue envelope

if compute_envelope
    for i_cond = 1:2
        cond = eval(['CONDITION' num2str(i_cond)]);
        cond_env = TRA_envelope(cond);
        eval(['CONDITION' num2str(i_cond) ' = cond_env;']);
    end        
end

%% downsample to 10 Hz

fs_ds = 10;
for i_cond = 1:2
    cond = eval(['CONDITION' num2str(i_cond)]);
    cond_ds = TRA_downsample(cond, fs_ds, extended_baseline);
    eval(['CONDITION' num2str(i_cond) ' = cond_ds;']);
end

n_samples = size(CONDITION1.sub1.ts, 2);

timevec = -2 : 1/fs_ds : 3;

%% format data for decoding analysis

% determine which conditions are to be decoded
switch i_decoding
    case 1
        % seen vs unseen
        newcond1 = [1];  
        newcond2 = [2];
        decoding_label = 'seen_v_unseen';
end


for i_sub = 1:n_subs 
  
    % z-score along sensors
    Xz_sensors = [];
    group = [];
    for i_cond = 1:2
        eval(['Xadd = CONDITION' int2str(i_cond) '.sub' int2str(i_sub) '.ts;']);
        
        Xz = zscore(Xadd,[],1); 
        Xz_sensors = cat(3, Xz_sensors, Xz);
        
        group = [group; i_cond*ones(size(Xadd,3),1)];
    end
    
    
    % regroup into 2 conditions
    regroup = zeros(size(group));
    for n = 1:length(newcond1)
        regroup(group==newcond1(n)) = 1; 
    end
    
    for n=1:length(newcond2)
        regroup(group==newcond2(n)) = -1; 
    end
    
    
    if sum(find(regroup==0))>0
        Xz_sensors(:,:,regroup==0)=[]; 
        regroup(regroup==0)=[];        
    end
    
    group = regroup;

    Xz_sensors_sub{i_sub} = Xz_sensors;
    
    group_sub{i_sub} = group;
end

%filename1 = ['./DecisionVariable_results/allsubs_conds.mat'];
%save(filename1, 'group_sub');

%% perform decoding analysis

for i_sub = 1:n_subs
    i_sub
    
    group = group_sub{i_sub};
    
    for i_sample = 1:n_samples
        
        % decode at current time sample for current subject
        data  = squeeze(Xz_sensors_sub{i_sub}(:, i_sample, :))';
        [svm_acc, models, activation_patterns, DVs] = libsvm_nfold_lk(group, data, n_folds, n_reps, libsvm_options_train, libsvm_options_test);

        % average across folds and sampling repetitions
        svm_acc_time(i_sub, i_sample) = mean(svm_acc(:));
        DVs_time{i_sub, i_sample} = DVs;
        
        % compute mean across-fold, across-rep activation pattern
        ind = 0;
        for i_fold = 1:n_folds
            for i_rep = 1:n_reps
                ind = ind + 1;
                
                ap(:, ind) = activation_patterns{i_fold, i_rep};
                w(:, ind)  = models{i_fold, i_rep}.SVs' * models{i_fold, i_rep}.sv_coef;
                b(ind)     = -models{i_fold, i_rep}.rho;
                
            end
        end
        
        ap_sub_time(:, i_sample, i_sub) = mean(ap, 2);
        w_sub_time(:, i_sample, i_sub) = mean(w, 2);
        b_sub_time(i_sub, i_sample) = mean(b);
        
    end
end

%% save data


filename = ['DecisionVariables.mat'];

save([data_dir filename], 'timevec', 'svm_acc_time', 'ap_sub_time', 'w_sub_time', 'b_sub_time', 'DVs_time');
