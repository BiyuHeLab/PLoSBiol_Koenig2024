#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 15:24:09 2021

@author: koenil04
"""

import os
import numpy as np
import pandas as pd
import mne
import math
from mne.time_frequency import (tfr_morlet,tfr_array_morlet)
from fooof import FOOOF
import scipy.io

os.chdir('/isilon/LFMI/VMdrive/Lua/MEG_alpha_1-f_SCP_PLoSBiol_2024/scripts')
data_dir = '/isilon/LFMI/VMdrive/Lua/MEG_alpha_1-f_SCP_PLoSBiol_2024/data/HL/'
figures_dir = '/isilon/LFMI/VMdrive/Lua/MEG_alpha_1-f_SCP_PLoSBiol_2024/figures/'
supp_dir = '/isilon/LFMI/VMdrive/Lua/MEG_alpha_1-f_SCP_PLoSBiol_2024/scripts/supporting_files_toolboxes/HL/'

import HLTP

def load_epochs(subject, filttype = None):
    if (filttype == 'raw'):
        epochs = mne.read_epochs(data_dir + 'Preprocessed/' + subject + '/HLTP_raw_stim-epo.fif')
    elif (filttype == 'filt'):
        epochs = mne.read_epochs(data_dir + 'Preprocessed/' + subject + '/HLTP_0_150_Hz_stim-epo.fif')
    epochs.pick_types(meg = True, ref_meg = False, exclude=[])
    return epochs

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

#Combine power data and behavioral data in order to run fooof according to conditions of interest (ie. recognized/unrecognized, correctly/incorrectly categorized)
#Load in behavioral data
all_df = pd.read_pickle(data_dir +'all_bhv_df.pkl')  
filttype = 'raw'
trialIDs_alpha = []

for s,subject in enumerate(HLTP.subjects): 
    print(subject)
    subset = (all_df.subject  == subject)
    behavdat = all_df[subset][['real_img','correct','recognition']]
    scrambled_origindex = behavdat[(behavdat.real_img == False)]
    scrambled_ind_origindex = scrambled_origindex.index
    trialIDs_alpha.append(np.array(scrambled_ind_origindex))
    
    behavdat.index = range(len(behavdat))
    scrambled = behavdat[(behavdat.real_img == False)]
    scrambled_ind = scrambled.index
    
    power = mne.time_frequency.read_tfrs(data_dir + 'Preprocessed/' + subject + '/power_spectrum_filt-tfr.h5')[0]
    power_trialIDs = power.selection
    trialIDs_scrambled = np.where(np.isin(power_trialIDs,scrambled_ind_origindex))
    psd = power.data #dimensions are trials x sensors x frequencies x timepoints
    psd = psd[scrambled_ind,:,:,3:39] #because the wavelet analysis won't produce reliable data at the start and end of the epochs
    
    if s == 4:
        insert_array = np.empty((psd.shape[0],psd.shape[2],psd.shape[3]))
        insert_array[:] = np.nan
        psd = np.insert(psd,74,insert_array,axis=1)
        psd = np.insert(psd,75,insert_array,axis=1)
    elif (s == 7) | (s == 18) | (s == 15):
        insert_array = np.empty((psd.shape[0],psd.shape[2],psd.shape[3]))
        insert_array[:] = np.nan
        psd = np.insert(psd,103,insert_array,axis=1)
    elif s == 8:
        insert_array = np.empty((psd.shape[0],psd.shape[2],psd.shape[3]))
        insert_array[:] = np.nan
        psd = np.insert(psd,241,insert_array,axis=1)
        psd = np.insert(psd,248,insert_array,axis=1)
    elif s == 20:
        insert_array = np.empty((psd.shape[0],psd.shape[2],psd.shape[3]))
        insert_array[:] = np.nan
        psd = np.insert(psd,56,insert_array,axis=1)
    elif s == 22:
        insert_array = np.empty((psd.shape[0],psd.shape[2],psd.shape[3]))
        insert_array[:] = np.nan
        psd = np.insert(psd,24,insert_array,axis=1)
        
    psd_400 = np.zeros((psd.shape[0],psd.shape[1],psd.shape[2],9)) #Will result in 10 400 ms intervals
    for time in range(9):
        psd_400[:,:,:,time] = np.nanmean(psd[:,:,:,(4*time):(4*time+4)],axis=3)
        
    n_trial = psd.shape[0]
    freqs = power.freqs
    alpha_BW = np.zeros((psd.shape[0],psd.shape[1],4)) #alpha bandwidth: trial x sensor x time
    alpha_CF = np.zeros((psd.shape[0],psd.shape[1],4)) #alpha center frequency
    beta_BW = np.zeros((psd.shape[0],psd.shape[1],4)) #beta bandwidth
    beta_CF = np.zeros((psd.shape[0],psd.shape[1],4)) #beta center frequency
    AUC_power = np.zeros((2,psd.shape[0],psd.shape[1],4)) #power computed using area under the curve, for alpha peak, 1/f in alpha band, total power in alpha band, beta peak, 1/f in beta band, total power in beta band
    
    for timepoint in range(4):
        print(timepoint)
        for sensor in range(272):
            for trial in range(n_trial):
                ps = psd_400[trial,sensor,:,timepoint] #This is a trial x frequencies power spectrum for the power averaged across the sensors in the cluster at the cluster's timepoint 
                if (any(np.isnan(ps))):
                    continue
                else:
                    fm = FOOOF(verbose=False,peak_threshold=1.0) #Initialize fooof group object
                    fm.fit(freqs, ps, [0,30]) #fit the fooof model between 0 and 30 Hz
                    #Find alpha peaks
                    if fm.peak_params_[(fm.peak_params_[:,0] > 7) & (fm.peak_params_[:,0] < 14),0].size == 0: #if there are no peaks in the alpha band
                        alpha_BW[trial,sensor,timepoint] = 0
                        alpha_CF[trial,sensor,timepoint] = np.nan
                    elif fm.peak_params_[(fm.peak_params_[:,0] > 7) & (fm.peak_params_[:,0] < 14),0].size == 1: #if there is exactly one peak in the alpha band
                        alpha_BW[trial,sensor,timepoint] = fm.peak_params_[(fm.peak_params_[:,0] > 7) & (fm.peak_params_[:,0] < 14),2]
                        alpha_CF[trial,sensor,timepoint] = fm.peak_params_[(fm.peak_params_[:,0] > 7) & (fm.peak_params_[:,0] < 14),0]
                    else: #if there is more than one peak in the alpha band, this selects the peak with the highest power
                        temp = fm.peak_params_[(fm.peak_params_[:,0] > 7) & (fm.peak_params_[:,0] < 14),:] #finds the peak in the alpha range with the highest power
                        alpha_BW[trial,sensor,timepoint] = temp[np.argmax(temp[:,1]),2]
                        alpha_CF[trial,sensor,timepoint] = temp[np.argmax(temp[:,1]),0]
                    #Find beta peaks
                    if fm.peak_params_[(fm.peak_params_[:,0] >= 14) & (fm.peak_params_[:,0] < 30),0].size == 0: #if there are no peaks in the beta band
                        beta_BW[trial,sensor,timepoint] = 0
                        beta_CF[trial,sensor,timepoint] = np.nan
                    elif fm.peak_params_[(fm.peak_params_[:,0] >= 14) & (fm.peak_params_[:,0] < 30),0].size == 1: #if there is exactly one peak in the beta band
                        beta_BW[trial,sensor,timepoint] = fm.peak_params_[(fm.peak_params_[:,0] >= 14) & (fm.peak_params_[:,0] < 30),2]
                        beta_CF[trial,sensor,timepoint] = fm.peak_params_[(fm.peak_params_[:,0] >= 14) & (fm.peak_params_[:,0] < 30),0]
                    else: #if there is more than one peak in the beta band, this selects the peak with the highest power
                        temp = fm.peak_params_[(fm.peak_params_[:,0] >= 14) & (fm.peak_params_[:,0] < 30),:] #finds the peak in the beta range with the highest power
                        beta_BW[trial,sensor,timepoint] = temp[np.argmax(temp[:,1]),2]
                        beta_CF[trial,sensor,timepoint] = temp[np.argmax(temp[:,1]),0]
                              
                    #Compute area under the curve for alpha and beta bands to extract power
                    linear_ap = 10**fm._ap_fit
                    linear_total = 10**fm.fooofed_spectrum_                    
                    AUC_ap_alpha = np.sum(linear_ap[int(np.where(fm.freqs==find_nearest(fm.freqs,alpha_CF[trial,sensor,timepoint]-alpha_BW[trial,sensor,timepoint]))[0]):int(np.where(fm.freqs==find_nearest(fm.freqs,alpha_CF[trial,sensor,timepoint]+alpha_BW[trial,sensor,timepoint]))[0])+1]) #Sums the ap fit between CF - BW and CF + BW (finding the nearest frequencies output by fm to those that correspond to the shaded green area of the peak)
                    AUC_total_alpha = np.sum(linear_total[int(np.where(fm.freqs==find_nearest(fm.freqs,alpha_CF[trial,sensor,timepoint]-alpha_BW[trial,sensor,timepoint]))[0]):int(np.where(fm.freqs==find_nearest(fm.freqs,alpha_CF[trial,sensor,timepoint]+alpha_BW[trial,sensor,timepoint]))[0])+1]) #Sums the total fooof fit between CF - BW and CF + BW (finding the nearest frequencies output by fm to those that correspond to the shaded green area of the peak)
                    AUC_ap_beta = np.sum(linear_ap[int(np.where(fm.freqs==find_nearest(fm.freqs,beta_CF[trial,sensor,timepoint]-beta_BW[trial,sensor,timepoint]))[0]):int(np.where(fm.freqs==find_nearest(fm.freqs,beta_CF[trial,sensor,timepoint]+beta_BW[trial,sensor,timepoint]))[0])+1]) #Sums the ap fit between CF - BW and CF + BW (finding the nearest frequencies output by fm to those that correspond to the shaded green area of the peak)
                    AUC_total_beta = np.sum(linear_total[int(np.where(fm.freqs==find_nearest(fm.freqs,beta_CF[trial,sensor,timepoint]-beta_BW[trial,sensor,timepoint]))[0]):int(np.where(fm.freqs==find_nearest(fm.freqs,beta_CF[trial,sensor,timepoint]+beta_BW[trial,sensor,timepoint]))[0])+1]) #Sums the total fooof fit between CF - BW and CF + BW (finding the nearest frequencies output by fm to those that correspond to the shaded green area of the peak)
                    if np.isnan(alpha_CF[trial,sensor,timepoint]):
                        AUC_power[0,trial,sensor,timepoint] = np.nan
                    else:
                        AUC_power[0,trial,sensor,timepoint] = AUC_total_alpha - AUC_ap_alpha
                    if np.isnan(beta_CF[trial,sensor,timepoint]):
                        AUC_power[1,trial,sensor,timepoint] = np.nan
                    else:
                        AUC_power[1,trial,sensor,timepoint] = AUC_total_beta - AUC_ap_beta
    np.save(data_dir + 'Preprocessed/' + subject + '/AUC_power_postfooof_scrambled.npy',AUC_power)
np.save(data_dir + 'trialIDs_scrambled.npy',trialIDs_alpha,allow_pickle=True)