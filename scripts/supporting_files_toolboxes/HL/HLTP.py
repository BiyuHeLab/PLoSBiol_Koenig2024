#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 14:29:14 2017

This file includes information specific to High-Level Threshold Perception 
MEG experiment and parameters for data processing. 

@author: podvalnye
"""
import scipy
import scipy.io as sio
import numpy as np
import pandas as pd
import mne
import pickle
import socket
import random
# Data directories

Project_dir = '/isilon/LFMI/archive/Projects/2019_NatComms_HLTP_MEG_Ella/'
    
MRI_dir = Project_dir + 'data/proc_data/freesurfer'
MEG_pro_dir = Project_dir + 'data/proc_data'
MEG_raw_dir = Project_dir + 'data/raw_data'
MEG_output_dir = '/isilon/LFMI/VMdrive/Lua/MEG_alpha_1-f_SCP_HLTP/data'

# Experiment and triggers
block_type = ['thresh',     # main experiment run - usually we have 10 blocks
              'localizer',  # visual localizer, images from 4 categories
              'rest',       # eyes open, fixation on a grey screen
              'quest'       # image threshold intensity estimation
              ]

stim_channel = 'UPPT002'       
PD_delay = 44 # samples, measures with photodiode, minimal delay is corrected
event_id = {"Stimulus": 1, "Trial_onset": 8,
            "Question1_category": 2, "Question2_experience": 4}

category_id = { "face": 1, "house": 2, "object": 3, "animal": 4, }
animacy_id = { "inanimate": -1, "animate": 1}

event_ids = {'Experience':{"Seen": 1, "Unseen": -1},
             'Accuracy': {"Correct": 1, "Incorrect": -1}, 
             'R_animacy':{"Animate": 1, "Inanimate": -1}, # R for response
             'S_animacy':{"Animate": 1, "Inanimate": -1}, # S for stimulus
             'R_category':category_id, 'S_category':category_id}   

# all subjects
subjects = ['AA', 'AC', 'AL', 'AR', 'AW', 'BJB', 'CW', 'DJ','EC', 'FSM', 'JA', 
            'JC', 'JP', 'JS', 'LS', 'NA', 'NC', 'NM', 'MC','SL', 'SM', 'SF', 
            'TL', 'TK'];

# subjects with mri            
mri_subj = {'good':['AA', 'AC', 'AL', 'AR', 'BJB', 'CW', 'DJ', 'EC', 'FSM','JA',
                    'JP', 'JS', 'LS', 'NA', 'NC', 'NM', 'MC', 'SM', 'TL', 'TK'],
            'bad':['SL', 'AW'],# these subjects mri cannot be used, bed bem 
            'no':['JC', 'SF']} # mri not available   

#TODO These subjects are contaminated by squid jumps          
jumps_subjects = [];        

# Bad channels 
bad_ch = [32, 172, 191] # these channels are missing from original CTF275 layout

# these channels defined bad by visual observation of localiser data
# (flat or huge artifacts)
bads= {'AW':['MLO52-1609','MLO53-1609'],
       'DJ':['MLT16-1609'],
       'EC':['MRT31-1609', 'MRT41-1609'],
       'MC':['MLT16-1609'],
       'NA':['MLT16-1609'],
       'SM':['MLF67-1609'],
       'TL':['MLC63-1609'],
       'AA':[],'AC':[],'AL':[],'AR':[], 'BJB':[], 'CW':[],'FSM':[], 'JA':[],
       'JP':[],'JS':[],'LS':[],'NC':[], 'NM':[], 'SL':[], 'TK':[], 'JC':[], 
       'SF':[]}

# Eye-tracking
pupil_chan = 304

random.seed(12121985)

#----- ICA parameters -----------------------------------------------

ica_decim = 3
ica_lo_freq, ica_hi_freq = 1, 45
ica_phase='zero-double' # this part makes difference in ICA components
ica_n_com = 50
ica_random_state = 23
ica_method = 'fastica'

# PreProcessing parameters
filter_phase = 'zero-double'
filter_length = '200ms'
filter_method = 'fir'

#----- Figures paramters ------------------------------------------------
hfont = {'fontname':'Helvetica'}
   
#----- Helper functions -------------------------------------------------------
                   
def save(var, file_name):
    outfile = open(file_name, 'wb')          
    pickle.dump(var, outfile)
    outfile.close()
            
def load(file_name):
    return pd.read_pickle(file_name)
    
def confusion_matrix(stim_cat, response_cat):
    
    conf_mat = np.zeros((len(category_id), len(category_id)))
    for s in category_id.values():
        R_given_S = response_cat[stim_cat == s] 
        n = R_given_S.shape[0]
        if n == 0: continue 
        for r in category_id.values():
            conf_mat[s - 1, r - 1] = sum(R_given_S == r) / float(n)
            
    return conf_mat

def confusion_matrix_raw(stim_cat, response_cat):
    
    conf_mat = np.zeros((len(category_id), len(category_id)))
    for s in category_id.values():
        R_given_S = response_cat[stim_cat == s] 
        n = R_given_S.shape[0]
        if n == 0: continue 
        for r in category_id.values():
            conf_mat[s - 1, r - 1] = sum(R_given_S == r) 
    return conf_mat

def bhv_measures(conf_mat):
    
    sensitivity = np.zeros(conf_mat.shape[0])
    specificity = np.zeros(conf_mat.shape[0])
    bias        = np.zeros(conf_mat.shape[0])
    
    for i in range(conf_mat.shape[0]):
        TP = conf_mat[i, i]
        FP = sum(conf_mat[:, i]) - TP
        FN = sum(conf_mat[i, :]) - TP
        TN = sum(sum(conf_mat[:, :])) - TP - FP - FN
        sensitivity[i] = TP/(TP + FN)
        specificity[i] = TN/(TN + FP)       
    bias = sum(conf_mat)
    
    return sensitivity, specificity, bias

# Get file name of meg raw data (i.e. *.ds) and return the block type
def get_block_type(filename):
    blockn = ''
    for b in block_type:
        idx = filename.find(b)
        if idx > 0:
            btype = b
            break
    if btype == 'thresh' or btype == 'rest':
        blockn = filename[-5:-3]
            
    btype =  btype + blockn
    return btype  
  
def get_block_names(subject):
    subj_raw_dir = MEG_raw_dir + '/' + subject + '/'
    filenames, _, _, _ = get_experimental_details(subj_raw_dir)
    block_names = []
    for f in filenames:
        block_names.append(get_block_type(f))
    return block_names

# needed for cluster tests, adjacency matrix
def get_connectivity():
    info = load('../supporting_files_toolboxes/info.p')
    connectivity, ch_names = mne.channels.read_ch_adjacency('ctf275')
    connectivity = connectivity.toarray()      
    pos = mne.find_layout(info).pos
    
    # remove bad channels:
    chn2remove = [];
    for bad_name in info['bads']:
        chn2remove.append(ch_names.index([s for s in ch_names if 
                                          bad_name[0:5] in s][0]))
    chn2remove= chn2remove + bad_ch
    chn2remove.sort()
    for k in chn2remove:
        pos = scipy.delete(pos, k, 0)
        connectivity = scipy.delete(connectivity, k, 0)
        connectivity = scipy.delete(connectivity, k, 1)
    connectivity = scipy.sparse.csr_matrix(connectivity)
    return connectivity, pos

# These are mostly remained from matlab, all should be fixed 
# Read details matlab struct, used historicaly, TODO: fix this separately
def get_experimental_details(subj_raw_dir):
    import os.path
    details = sio.loadmat(subj_raw_dir  + 'details.mat')['details']    
    filenames = details['block_file'][0,0]
    date = details['date'][0,0][0]
    subj_code = details['subj_code'][0,0][0]     
    subj_meg_dir = subj_raw_dir + 'MEG/' + subj_code  
     
    raw_filenames = [];
    for b in range(0, filenames.shape[1]):   
        raw_filenames.append(subj_meg_dir + filenames[0, b][0] + '.ds')        
    rest_run1 = subj_meg_dir + '_rest_' + date + '_01.ds'
    raw_filenames.append(rest_run1);
    rest_run2 = subj_meg_dir + '_rest_' + date + '_02.ds' 
    if os.path.exists(rest_run2):     
        raw_filenames.append(rest_run2)
    n_blocks = details['block_n'][0,0][0][0]
    return raw_filenames, subj_code, n_blocks, date

# Read epochs, remove reference channels, interpolate bad channels, remove bad trials
def get_raw_epochs(s, epoch_name):
    epochs = mne.read_epochs(MEG_pro_dir + '/' + s + '/' + epoch_name + 
                             '-epo.fif')
    epochs.pick_types(meg = True, ref_meg = False, exclude=[])
    
    if len(epochs.info['bads']):
        epochs.interpolate_bads(reset_bads = False,origin=(0., 0., 0.04))
    #epochs.drop_bad(reject=dict(mag = 4e-12))
    return epochs

def get_selected_epochs(epochs, events, event_id):
    out_epochs = epochs.copy()
    out_epochs.events[:,2] = events[out_epochs.selection]
    out_epochs.event_id = event_id
    out_epochs.drop(np.where(out_epochs.events[:,2]==0)[0])
    return out_epochs

def get_ROI(s, roi):
    
    ch_names = load(MEG_pro_dir + '/results/meg_chan_names.p')
    if roi == "HV":
        #sensors = sio.loadmat(MEG_pro_dir + subjects[s]
        #    +'/results/loc_animacy_sensors.mat')['all_sensors'][0]
       sensors = load(MEG_pro_dir +'/' + s + '/loc_animacy_sensors_clean.p')
       
       if len(sensors):
           sensors = sensors.astype(int)
    elif roi == "SU":
        sensors = load(MEG_pro_dir  + '/results/SU_sensors.p')[0]
        sensors = sensors.astype(int)
    elif roi == "A":
        sensors =  [i for i, ss in enumerate(ch_names)]
    else:
        sensors = [i for i, ss in enumerate(ch_names) if 
                   ('MR'+roi in ss) or ('ML'+roi in ss) or ('MZ' + roi in ss)]
    chan_to_select = [ch_names[i] for i in sensors]
    return chan_to_select, sensors

def get_td_coef(td):
    from mne.decoding import get_coef
    n_chan = len(td.ch_names)
    n_time = len(td.estimators_)
    coefs = np.zeros((n_chan, n_time))
    for t in range(n_time):
        t_coef = []
        for fold in range(td.cv_.n_splits):
            curr_est = td.estimators_[t][fold]           
            fold_coef = get_coef(curr_est, attr='patterns_', 
                                 inverse_transform = True)
            t_coef.append(fold_coef)            
        coefs[:, t] = np.squeeze(np.dstack(t_coef)).mean(axis=1)    
    return coefs


#------ should be removed----------------------------------------
#TODO verify the code doesn't use the functions below, 
# work with data frames intead
def get_behavior(subject):
    subj_raw_dir = MEG_raw_dir + '/' + subject + '/'
    bhv = sio.loadmat(subj_raw_dir  + '/Behavior/bhv_data.mat')['bhv_data']    
    seen = bhv['seen'][0,0][0]
    unseen = bhv['unseen'][0,0][0]
    real = bhv['real_img'][0,0][0]
    cat_response = bhv['cat_response'][0,0][0]
    cat_protocol = bhv['cat_protocol'][0,0][0]
    return seen, unseen, cat_response, cat_protocol, real

def get_exemplar(subject):
    subj_raw_dir = MEG_raw_dir + '/' + subject + '/'
    bhv = sio.loadmat(subj_raw_dir  + '/Behavior/datafile.mat')['data']
    exemplars = bhv['exemplar'][0,0][0]
    stimid = bhv['stimID'][0,0][0]
    return exemplars, stimid

def get_labels(label_name = '', label_subset = {}):
    #TODO check if label is available and print all options
    all_df = pd.read_pickle(MEG_pro_dir +'/results/all_bhv_df.pkl')   
    labels = []
    for subject in subjects:
        subj_labels = np.zeros(360)
        subset = (all_df.subject  == subject)
        for k in label_subset.keys():
            subset = subset & (all_df[k] == label_subset[k])   
        bhv = all_df[subset][label_name]
        # For exemplars, translate to integers so it works with other parts
        #if bhv.get_values()[0].dtype != np.dtype('uint8'):
        #    for idx in range(len(bhv)):
        #        cat_name = bhv.get_values()[idx].astype('str')[:-1]
        #        exemplar_num = int(bhv.get_values()[idx][-1:])
        #        subj_labels[bhv.index[idx]] = category_id[cat_name] * 10 + exemplar_num
        #elif bhv.get_values()[0].dtype == np.dtype('uint8'):     
        #    subj_labels[bhv.index] = bhv.get_values()    
        subj_labels[bhv.index] = bhv.to_numpy()    
        labels.append(subj_labels)
    return labels

def prep_data_and_label(input_data, trial_n, label_name, label_subset, 
                        label_values):
    data = []
    tr_return = []
    labels = get_labels(label_name, label_subset)
    for sub_idx, _ in enumerate(subjects):
        labels[sub_idx] =  labels[sub_idx][trial_n[sub_idx]]
        tr_to_reject = np.where(np.logical_not(
                [k in label_values for k in labels[sub_idx]]))
        data.append(np.delete(input_data[sub_idx], tr_to_reject, axis = 0))
        labels[sub_idx] = np.delete(labels[sub_idx], tr_to_reject)
        tr_return.append(np.delete(trial_n[sub_idx].copy(), tr_to_reject, axis = 0))
    return data, labels, tr_return 


