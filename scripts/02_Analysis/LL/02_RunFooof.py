#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 12:11:52 2021

@author: koenil04
"""

import numpy as np
from scipy.io import loadmat, savemat
import mat73
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import math
from scipy import stats

from fooof import FOOOF
import os

os.chdir('/isilon/LFMI/VMdrive/Lua/MEG_alpha_1-f_SCP_PLoSBiol_2024/scripts')
data_dir = '/isilon/LFMI/VMdrive/Lua/MEG_alpha_1-f_SCP_PLoSBiol_2024/data/LL/'
figures_dir = '/isilon/LFMI/VMdrive/Lua/MEG_alpha_1-f_SCP_PLoSBiol_2024/figures/'
supp_dir = '/isilon/LFMI/VMdrive/Lua/MEG_alpha_1-f_SCP_PLoSBiol_2024/scripts/supporting_files_toolboxes/LL/'

conds = ['seen','unseen']

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

Posc_all = [] #List of two conditions per subject (eg. sub1 cond1, sub1 cond2, sub2cond1, etc). Each element contains a trials x time (45) x sensor (273) array.

for sub in range(1,12):
    for cond in range(0,2):
        print(sub)
        print(cond)
        file_path = data_dir + 'Preprocessed/' + 'power_' + str(sub) + '_' + conds[cond] + '.mat'
        data = mat73.loadmat(file_path) # Load the mat v7.3 file 
        freqs = data['TFR']['freq'] #unpack data from dictionary
        psd = data['TFR']['powspctrm'][:,:,:,3:38] #this has power values for the dimensions trials x sensors x frequencies x times
        psd_400 = np.zeros((psd.shape[0],psd.shape[1],psd.shape[2],9)) #Will result in 10 400 ms intervals
        for time in range(9):
            psd_400[:,:,:,time] = np.nanmean(psd[:,:,:,(4*time):(4*time+4)],axis=3)
        
        n_trial = psd_400.shape[0]
        n_sensors = psd_400.shape[1]
        n_times = psd_400.shape[3]
        results = np.zeros((10,n_trial,n_times,n_sensors)) #(aperiodic exponent, aperiodic offset, alpha peak frequency, alpha peak power, alpha bandwidth, aperiodic height at alpha CF, beta peak frequency, beta peak power, beta bandwidth, aperiodic height at beta CF) x (trials) x (timepoints) x (sensors)
        Posc_auc = np.zeros((6, n_trial,n_times,n_sensors)) #Compute power using AUC method: (alpha AUC, 1/f AUC alpha, total AUC alpha, beta AUC, 1/f AUC beta, total AUC beta) x trials x times x sensors
        for sensor in range(0,n_sensors):
            for trial in range(0,n_trial):
                print('sub = ' + str(sub) + '; cond = ' + str(cond) + '; sensor = ' + str(sensor) + '; trial = ' + str(trial))
                for timepoint in range(9): #all pre- to post-stim
                    ps = psd_400[trial,sensor,:,timepoint]
                    na_freqs = np.isnan(ps) #Remove na values for low frequencies that originate from the wavelet parameters
                    ps_trial = np.delete(ps,na_freqs)
                    freqs_trial = np.delete(freqs,na_freqs)
                    
                    fm = FOOOF(verbose=False,peak_threshold=1.0) # Initialize FOOOF object
                    fm.fit(freqs_trial, ps_trial, [0, 30]) # Fit the FOOOF model, and report
                    if fm.has_model:                    
                        results[0,trial,timepoint,sensor] = fm.get_params('aperiodic_params', 'exponent') #aperiodic exponent
                        results[1,trial,timepoint,sensor] = fm.get_params('aperiodic_params', 'offset') #aperiodic offset
                        #Find alpha parameters
                        if fm.peak_params_[(fm.peak_params_[:,0] > 7) & (fm.peak_params_[:,0] < 14),0].size == 0: #if there are no peaks in the alpha band
                            results[2,trial,timepoint,sensor] = np.nan
                            results[3,trial,timepoint,sensor] = np.nan
                            results[4,trial,timepoint,sensor] = np.nan
                            results[5,trial,timepoint,sensor] = np.nan
                        elif fm.peak_params_[(fm.peak_params_[:,0] > 7) & (fm.peak_params_[:,0] < 14),0].size == 1: #if there is exactly one peak in the alpha band
                            results[2,trial,timepoint,sensor] = fm.peak_params_[(fm.peak_params_[:,0] > 7) & (fm.peak_params_[:,0] < 14),0] #peak alpha frequency
                            results[3,trial,timepoint,sensor] = fm.peak_params_[(fm.peak_params_[:,0] > 7) & (fm.peak_params_[:,0] < 14),1] #peak alpha power
                            results[4,trial,timepoint,sensor] = fm.peak_params_[(fm.peak_params_[:,0] > 7) & (fm.peak_params_[:,0] < 14),2] #peak alpha bandwidth
                            results[5,trial,timepoint,sensor] = fm._ap_fit[np.where(freqs_trial==find_nearest(freqs_trial,results[2,trial,timepoint,sensor]))] #height of the 1/f curve at the alpha center frequency
                        else : #if there is more than one peak in the alpha band, this selects the peak with the highest power
                            temp = fm.peak_params_[(fm.peak_params_[:,0] > 7) & (fm.peak_params_[:,0] < 14),:] #finds the peak in the alpha range with the highest power
                            results[2,trial,timepoint,sensor] = temp[np.argmax(temp[:,1]),0]
                            results[3,trial,timepoint,sensor] = temp[np.argmax(temp[:,1]),1]
                            results[4,trial,timepoint,sensor] = temp[np.argmax(temp[:,1]),2]
                            results[5,trial,timepoint,sensor] = fm._ap_fit[np.where(freqs_trial==find_nearest(freqs_trial,results[2,trial,timepoint,sensor]))] #height of the 1/f curve at the alpha center frequency
                        linear_ap = 10**fm._ap_fit
                        linear_total = 10**fm.fooofed_spectrum_
                        AUC_ap = np.sum(linear_ap[int(np.where(fm.freqs==find_nearest(fm.freqs,results[2,trial,timepoint,sensor]-results[4,trial,timepoint,sensor]))[0]):int(np.where(fm.freqs==find_nearest(fm.freqs,results[2,trial,timepoint,sensor]+results[4,trial,timepoint,sensor]))[0])+1]) #Sums the ap fit between CF - BW and CF + BW (finding the nearest frequencies output by fm to those that correspond to the shaded green area of the peak)
                        AUC_total = np.sum(linear_total[int(np.where(fm.freqs==find_nearest(fm.freqs,results[2,trial,timepoint,sensor]-results[4,trial,timepoint,sensor]))[0]):int(np.where(fm.freqs==find_nearest(fm.freqs,results[2,trial,timepoint,sensor]+results[4,trial,timepoint,sensor]))[0])+1])
                        if np.isnan(results[2,trial,timepoint,sensor]):
                            Posc_auc[0,trial,timepoint,sensor] = np.nan
                        else:    
                            Posc_auc[0,trial,timepoint,sensor] = AUC_total - AUC_ap
                        Posc_auc[1,trial,timepoint,sensor] = AUC_ap
                        Posc_auc[2,trial,timepoint,sensor] = AUC_total
                        ###Compute beta oscillation AUC
                        if fm.peak_params_[(fm.peak_params_[:,0] >= 14) & (fm.peak_params_[:,0] < 30),0].size == 0: #if there are no peaks in the beta band
                            results[6,trial,timepoint,sensor] = np.nan
                            results[7,trial,timepoint,sensor] = np.nan
                            results[8,trial,timepoint,sensor] = np.nan
                            results[9,trial,timepoint,sensor] = np.nan
                        elif fm.peak_params_[(fm.peak_params_[:,0] >= 14) & (fm.peak_params_[:,0] < 30),0].size == 1: #if there is exactly one peak in the beta band
                            results[6,trial,timepoint,sensor] = fm.peak_params_[(fm.peak_params_[:,0] >= 14) & (fm.peak_params_[:,0] < 30),0] #peak beta frequency
                            results[7,trial,timepoint,sensor] = fm.peak_params_[(fm.peak_params_[:,0] >= 14) & (fm.peak_params_[:,0] < 30),1] #peak beta power
                            results[8,trial,timepoint,sensor] = fm.peak_params_[(fm.peak_params_[:,0] >= 14) & (fm.peak_params_[:,0] < 30),2] #peak beta bandwidth
                            results[9,trial,timepoint,sensor] = fm._ap_fit[np.where(freqs_trial==find_nearest(freqs_trial,results[6,trial,timepoint,sensor]))] #height of the 1/f curve at the beta center frequency
                        else : #if there is more than one peak in the alpha band, this selects the peak with the highest power
                            temp = fm.peak_params_[(fm.peak_params_[:,0] >= 14) & (fm.peak_params_[:,0] < 30),:] #finds the peak in the beta range with the highest power
                            results[6,trial,timepoint,sensor] = temp[np.argmax(temp[:,1]),0]
                            results[7,trial,timepoint,sensor] = temp[np.argmax(temp[:,1]),1]
                            results[8,trial,timepoint,sensor] = temp[np.argmax(temp[:,1]),2]
                            results[9,trial,timepoint,sensor] = fm._ap_fit[np.where(freqs_trial==find_nearest(freqs_trial,results[6,trial,timepoint,sensor]))] #height of the 1/f curve at the alpha center frequency
                        AUC_ap2 = np.sum(linear_ap[int(np.where(fm.freqs==find_nearest(fm.freqs,results[6,trial,timepoint,sensor]-results[8,trial,timepoint,sensor]))[0]):int(np.where(fm.freqs==find_nearest(fm.freqs,results[6,trial,timepoint,sensor]+results[8,trial,timepoint,sensor]))[0])+1]) #Sums the ap fit between CF - BW and CF + BW (finding the nearest frequencies output by fm to those that correspond to the shaded green area of the peak)
                        AUC_total2 = np.sum(linear_total[int(np.where(fm.freqs==find_nearest(fm.freqs,results[6,trial,timepoint,sensor]-results[8,trial,timepoint,sensor]))[0]):int(np.where(fm.freqs==find_nearest(fm.freqs,results[6,trial,timepoint,sensor]+results[8,trial,timepoint,sensor]))[0])+1])
                        if np.isnan(results[6,trial,timepoint,sensor]):
                            Posc_auc[3,trial,timepoint,sensor] = np.nan
                        else:    
                            Posc_auc[3,trial,timepoint,sensor] = AUC_total2 - AUC_ap2
                        Posc_auc[4,trial,timepoint,sensor] = AUC_ap2
                        Posc_auc[5,trial,timepoint,sensor] = AUC_total2
                    else:
                        results[:,trial,timepoint,sensor] = np.nan
                        Posc_auc[:,trial,timepoint,sensor] = np.nan
        Posc_all.append(Posc_auc)
Posc_all_dict = np.empty((len(Posc_all),), dtype=np.object)
for i in range(len(Posc_all)):
    Posc_all_dict[i] = Posc_all[i]
filename2 = data_dir + 'allpostfooofpower.mat'
savemat(filename2, {"Posc_all_dict":Posc_all_dict})
np.save(data_dir + 'allpostfooofpower.npy',Posc_all_dict)