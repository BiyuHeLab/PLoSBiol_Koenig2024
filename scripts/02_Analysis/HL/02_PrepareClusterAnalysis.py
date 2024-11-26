#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 16:10:08 2021

@author: koenil04
"""
import os
import numpy as np
import pandas as pd
import mne
import scipy.stats
from scipy.io import savemat

os.chdir('/isilon/LFMI/VMdrive/Lua/MEG_alpha_1-f_SCP_PLoSBiol_2024/scripts')
data_dir = '/isilon/LFMI/VMdrive/Lua/MEG_alpha_1-f_SCP_PLoSBiol_2024/data/HL/'
figures_dir = '/isilon/LFMI/VMdrive/Lua/MEG_alpha_1-f_SCP_PLoSBiol_2024/figures/'
supp_dir = '/isilon/LFMI/VMdrive/Lua/MEG_alpha_1-f_SCP_PLoSBiol_2024/scripts/supporting_files_toolboxes/HL/'

import HLTP

def load(file_name):
    return pd.read_pickle(file_name)

#Import behavioral data in order to separate trials by recognized/unrecognized
all_df = pd.read_pickle(data_dir +'all_bhv_df.pkl')  

alpha_persub = [] #List of length nSubs of numpy arrays of the shape nTrials x nSensors x nTimepoints
beta_persub = [] #List of length nSubs of numpy arrays of the shape nTrials x nSensors x nTimepoints
alpha_matlab = []
beta_matlab = []
alpha_matlab_log = []
beta_matlab_log = []
alpha_matlab_acc = []
beta_matlab_acc = []
trialIDs_alpha = []

nTrials_conds = [] #number of trials for seen, unseen, for each subject

#Huge artifacts in certain trials for subject 16 -- remove trials
remove_seen_s16 = np.array([41])
remove_unseen_s16 = np.array([72,113,137,149,156,160,167,195])

percenttrials_remaining_alpha = np.zeros((len(HLTP.subjects),2)) #for each subject, first count of how many trials total, and then after removing mean average of trials with no peaks (across sensors and timepoints), compute percent of original trials remaining
percenttrials_remaining_beta = np.zeros((len(HLTP.subjects),2)) #for each subject, first count of how many trials total, and then after removing mean average of trials with no peaks (across sensors and timepoints), compute percent of original trials remaining

for s, subject in enumerate(HLTP.subjects):
    #Select subject-specific data
    subset = (all_df.subject  == subject)
    behavdat = all_df[subset][['cat_protocol','real_img','correct','recognition']]
    seen_origindex = behavdat[(behavdat.real_img == True) & (behavdat.recognition == 1)].index
    unseen_origindex = behavdat[(behavdat.real_img == True) & (behavdat.recognition == -1)].index
    
    if s == 16:
        seen_origindex = np.delete(seen_origindex,remove_seen_s16)
        unseen_origindex = np.delete(unseen_origindex,remove_unseen_s16)
    nTrials_conds.append([len(seen_origindex),len(unseen_origindex)])
    
    #Import alpha and beta power, format: trials x sensors x timepoints (-1.7 to +1.6 in 100ms increments)
    original_power = mne.time_frequency.read_tfrs(data_dir + 'Preprocessed/' + subject + '/power_spectrum_raw_4timewindows-tfr.h5')[0]
    power_trialIDs = original_power.selection
    alpha_power = np.load(data_dir + 'Preprocessed/' + subject + '/AUC_power_postfooof.npy')[0,:,:,:]
    beta_power = np.load(data_dir + 'Preprocessed/' + subject + '/AUC_power_postfooof.npy')[3,:,:,:]
    
    alpha_4 = alpha_power[:,:,:4]
    beta_4 = beta_power[:,:,:4]
    
    alpha_4_seen = alpha_4[np.isin(power_trialIDs,seen_origindex),:,:]
    alpha_4_unseen = alpha_4[np.isin(power_trialIDs,unseen_origindex),:,:]
    trialIDs_seen = power_trialIDs[np.isin(power_trialIDs, seen_origindex)]
    trialIDs_unseen = power_trialIDs[np.isin(power_trialIDs,unseen_origindex)]
    trialIDs_real = np.concatenate((trialIDs_seen,trialIDs_unseen))
    alpha_real = np.concatenate((alpha_4_seen,alpha_4_unseen),axis=0)
    alpha_persub.append(alpha_real)
    trialIDs_alpha.append(trialIDs_real)
    
    alpha_matlab.append(alpha_4_seen)
    alpha_matlab.append(alpha_4_unseen)
    
    beta_4_seen = beta_4[np.isin(power_trialIDs,seen_origindex),:,:]
    beta_4_unseen = beta_4[np.isin(power_trialIDs,unseen_origindex),:,:]
    beta_real = np.concatenate((beta_4_seen,beta_4_unseen),axis=0)
    beta_persub.append(beta_real)
    
    beta_matlab.append(beta_4_seen)
    beta_matlab.append(beta_4_unseen)
    
np.save(data_dir + 'alpha_persubject.npy',alpha_matlab,allow_pickle=True)
np.save(data_dir + 'beta_persubject.npy',beta_matlab,allow_pickle=True)
np.save(data_dir + 'trialIDs_real.npy',trialIDs_alpha,allow_pickle=True)
np.save(data_dir + 'alpha_alltrials.npy',alpha_persub,allow_pickle=True)
np.save(data_dir + 'beta_alltrials.npy',beta_persub,allow_pickle=True)

#Save as .mat for cluster analysis
alpha_persubdict = np.empty((len(alpha_matlab),), dtype=np.object)
for i in range(len(alpha_matlab)):
    alpha_persubdict[i] = alpha_matlab[i]
filename = data_dir + 'alpha_per_subject.mat'
savemat(filename, {"power":alpha_persubdict})    

beta_persubdict = np.empty((len(beta_matlab),), dtype=np.object)
for i in range(len(beta_matlab)):
    beta_persubdict[i] = beta_matlab[i]
filename2 = data_dir + 'beta_per_subject.mat'
savemat(filename2, {"power":beta_persubdict}) 

