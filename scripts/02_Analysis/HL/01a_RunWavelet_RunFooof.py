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
from fooof import FOOOFGroup
from fooof import FOOOF
import multiprocessing as mp
import sys

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

def output_power(subject,filttype):
     #Load in data
     epochs = load_epochs(subject,filttype) #sampling freq. = 1200 Hz
     epochs_shorter = epochs.copy().crop(tmin=-1.7)
     #Set up Wavelet parameters
     print(subject)
     print(filttype)
     freqs = np.arange(0,40,0.8)[3:]
     n_cycles = np.linspace(3,9,freqs.shape[0])
     power = tfr_morlet(epochs_shorter, freqs=freqs,n_cycles=n_cycles, return_itc=False, average=False,decim=480) #decim = 120 downsamples the 1200 Hz data to obtain power estimates for 10 samples per second (40 samples across 4 sec of data; speeds up the computation considerably)
     file_name = 'power_spectrum_' + filttype +'_4timewindows'
     power.save(data_dir + 'Preprocessed/' + subject + '/' + file_name + '-tfr.h5',overwrite=True) 

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

#Combine power data and behavioral data in order to run fooof according to conditions of interest (ie. recognized/unrecognized, correctly/incorrectly categorized)
#Load in behavioral data
all_df = pd.read_pickle(data_dir +'all_bhv_df.pkl')  

def run_fooof(subject,use_400): #subject is string with subject ID, use_400 is True/False on whether to use 400 ms averages of power
    power = mne.time_frequency.read_tfrs(data_dir + 'Preprocessed/' + subject + '/power_spectrum_raw_4timewindows-tfr.h5')[0]
    psd = power.data[:,:,:,3:39] #dimensions are trials x sensors x frequencies x timepoints
    psd_400 = np.zeros((psd.shape[0],psd.shape[1],psd.shape[2],9)) #Will result in 10 400 ms intervals
    for time in range(9):
        psd_400[:,:,:,time] = np.nanmean(psd[:,:,:,(4*time):(4*time+4)],axis=3)
    
    if use_400:
        psd = psd_400
        
    freqs = power.freqs
    #psd_stacked = np.vstack(psd) #this collapses trials and sensors into one column
    ap_exps = np.zeros((psd.shape[0],psd.shape[1],psd.shape[3])) #will contain 1/f exponent for each trial, sensor and timepoint
    alpha_BW = np.zeros((psd.shape[0],psd.shape[1],psd.shape[3])) #alpha bandwidth
    alpha_CF = np.zeros((psd.shape[0],psd.shape[1],psd.shape[3])) #alpha center frequency
    beta_BW = np.zeros((psd.shape[0],psd.shape[1],psd.shape[3])) #beta bandwidth
    beta_CF = np.zeros((psd.shape[0],psd.shape[1],psd.shape[3])) #beta center frequency
    AUC_power = np.zeros((6,psd.shape[0],psd.shape[1],psd.shape[3])) #power computed using area under the curve, for alpha peak, 1/f in alpha band, total power in alpha band, beta peak, 1/f in beta band, total power in beta band
    trialNs = np.zeros((4,psd.shape[3],psd.shape[1]))
    
    for timepoint in range(psd.shape[3]): #-1.7 sec to +1.6 sec
        print(subject)
        print(timepoint)
        for sensor in range(0,psd.shape[1]):
            for trial in range(0,psd.shape[0]):
                ps = psd[trial,sensor,:,timepoint]
                fm = FOOOF(verbose=False,peak_threshold=1.0) # Initialize FOOOF object
                fm.fit(freqs, ps, [0, 30]) # Fit the FOOOF model, and report
                #fm.plot()
                if fm.has_model:
                    ap_exps[trial,sensor,timepoint] = fm.get_params('aperiodic_params', 'exponent') #aperiodic exponent
                    #Find alpha peaks
                    if fm.peak_params_[(fm.peak_params_[:,0] >= 7) & (fm.peak_params_[:,0] < 14),0].size == 0: #if there are no peaks in the alpha band
                        alpha_BW[trial,sensor,timepoint] = 0
                        alpha_CF[trial,sensor,timepoint] = np.nan
                    elif fm.peak_params_[(fm.peak_params_[:,0] >= 7) & (fm.peak_params_[:,0] < 14),0].size == 1: #if there is exactly one peak in the alpha band
                        alpha_BW[trial,sensor,timepoint] = fm.peak_params_[(fm.peak_params_[:,0] >= 7) & (fm.peak_params_[:,0] < 14),2]
                        alpha_CF[trial,sensor,timepoint] = fm.peak_params_[(fm.peak_params_[:,0] >= 7) & (fm.peak_params_[:,0] < 14),0]
                    else: #if there is more than one peak in the alpha band, this selects the peak with the highest power
                        temp = fm.peak_params_[(fm.peak_params_[:,0] >= 7) & (fm.peak_params_[:,0] < 14),:] #finds the peak in the alpha range with the highest power
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
                    
                    #Compute number of trials with both alpha and beta peaks, only alpha peaks, only beta peaks, or neither.
                    if (fm.peak_params_[(fm.peak_params_[:,0] >= 7) & (fm.peak_params_[:,0] < 14),:].size >= 1) and (fm.peak_params_[(fm.peak_params_[:,0] >= 14) & (fm.peak_params_[:,0] < 30),:].size >= 1): #if there is at least one alpha and one beta peak in this trial
                        trialNs[0,timepoint,sensor] = trialNs[0,timepoint,sensor]+1
                    elif (fm.peak_params_[(fm.peak_params_[:,0] >= 7) & (fm.peak_params_[:,0] < 14),:].size >= 1) and (fm.peak_params_[(fm.peak_params_[:,0] >= 14) & (fm.peak_params_[:,0] < 30),:].size == 0): #If there is only an alpha peak
                        trialNs[1,timepoint,sensor] = trialNs[1,timepoint,sensor]+1
                    elif (fm.peak_params_[(fm.peak_params_[:,0] >= 7) & (fm.peak_params_[:,0] < 14),:].size == 0) and (fm.peak_params_[(fm.peak_params_[:,0] >= 14) & (fm.peak_params_[:,0] < 30),:].size >= 1): #If there is only a beta peak
                        trialNs[2,timepoint,sensor] = trialNs[2,timepoint,sensor]+1
                    else : #if there is neither an alpha nor a beta peak
                        trialNs[3,timepoint,sensor] =  trialNs[3,timepoint,sensor]+1
                        
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
                    AUC_power[1,trial,sensor,timepoint] = AUC_ap_alpha
                    AUC_power[2,trial,sensor,timepoint] = AUC_total_alpha
                    if np.isnan(beta_CF[trial,sensor,timepoint]):
                        AUC_power[3,trial,sensor,timepoint] = np.nan
                    else:
                        AUC_power[3,trial,sensor,timepoint] = AUC_total_beta - AUC_ap_beta
                    AUC_power[4,trial,sensor,timepoint] = AUC_ap_beta
                    AUC_power[5,trial,sensor,timepoint] = AUC_total_beta
                else:
                    ap_exps[trial,sensor,timepoint] = np.nan
                    alpha_BW[trial,sensor,timepoint] = np.nan
                    alpha_CF[trial,sensor,timepoint] = np.nan
                    beta_BW[trial,sensor,timepoint] = np.nan
                    beta_CF[trial,sensor,timepoint] = np.nan
                    AUC_power[:,trial,sensor,timepoint] = np.nan
    np.save('../data/' + subject + '/AUC_power_postfooof.npy',AUC_power)
    
if __name__ == '__main__':
    globals()[sys.argv[1]](*sys.argv[2:])

# #Some subjects don't have all their sensors:
# bads= {'AW':[74,75],
#        'DJ':[103],
#        'EC':[241, 248],
#        'MC':[103],
#        'NA':[103],
#        'SM':[56],
#        'TL':[24],}        
# #Subject 'AW' has two sensors (74 and 75) with no data. Append these channels and fill them with NA, to keep them similar in structure to other subjects
# all_power = np.load('../data/AW/AUC_power_postfooof_' + filttype + '.npy')
# empty_array_for_insertion = np.empty((all_power.shape[0],all_power.shape[1],all_power.shape[3]))
# empty_array_for_insertion[:] = math.nan
# all_power = np.insert(all_power,bads['AW'][1],empty_array_for_insertion,axis=2)
# all_power = np.insert(all_power,bads['AW'][0],empty_array_for_insertion,axis=2)
# np.save('../data/AW/AUC_power_postfooof_' + filttype + '.npy',all_power)

# all_power = np.load('../data/DJ/AUC_power_postfooof_' + filttype + '.npy')
# empty_array_for_insertion = np.empty((all_power.shape[0],all_power.shape[1],all_power.shape[3]))
# empty_array_for_insertion[:] = math.nan
# all_power = np.insert(all_power,bads['DJ'][0],empty_array_for_insertion,axis=2)
# np.save('../data/DJ/AUC_power_postfooof_' + filttype + '.npy',all_power)

# all_power = np.load('../data/EC/AUC_power_postfooof_' + filttype + '.npy')
# empty_array_for_insertion = np.empty((all_power.shape[0],all_power.shape[1],all_power.shape[3]))
# empty_array_for_insertion[:] = math.nan
# all_power = np.insert(all_power,bads['EC'][1],empty_array_for_insertion,axis=2)
# all_power = np.insert(all_power,bads['EC'][0],empty_array_for_insertion,axis=2)
# np.save('../data/EC/AUC_power_postfooof_' + filttype + '.npy',all_power)

# all_power = np.load('../data/MC/AUC_power_postfooof_' + filttype + '.npy')
# empty_array_for_insertion = np.empty((all_power.shape[0],all_power.shape[1],all_power.shape[3]))
# empty_array_for_insertion[:] = math.nan
# all_power = np.insert(all_power,bads['MC'][0],empty_array_for_insertion,axis=2)
# np.save('../data/MC/AUC_power_postfooof_' + filttype + '.npy',all_power)

# all_power = np.load('../data/NA/AUC_power_postfooof_' + filttype + '.npy')
# empty_array_for_insertion = np.empty((all_power.shape[0],all_power.shape[1],all_power.shape[3]))
# empty_array_for_insertion[:] = math.nan
# all_power = np.insert(all_power,bads['NA'][0],empty_array_for_insertion,axis=2)
# np.save('../data/NA/AUC_power_postfooof_' + filttype + '.npy',all_power)

# all_power = np.load('../data/SM/AUC_power_postfooof_' + filttype + '.npy')
# empty_array_for_insertion = np.empty((all_power.shape[0],all_power.shape[1],all_power.shape[3]))
# empty_array_for_insertion[:] = math.nan
# all_power = np.insert(all_power,bads['SM'][0],empty_array_for_insertion,axis=2)
# np.save('../data/SM/AUC_power_postfooof_' + filttype + '.npy',all_power)

# all_power = np.load('../data/TL/AUC_power_postfooof_' + filttype + '.npy')
# empty_array_for_insertion = np.empty((all_power.shape[0],all_power.shape[1],all_power.shape[3]))
# empty_array_for_insertion[:] = math.nan
# all_power = np.insert(all_power,bads['TL'][0],empty_array_for_insertion,axis=2)
# np.save('../data/TL/AUC_power_postfooof_' + filttype + '.npy',all_power)
