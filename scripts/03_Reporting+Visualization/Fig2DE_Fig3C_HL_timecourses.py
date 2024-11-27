#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 14:56:09 2022

@author: koenil04
"""

import os
import numpy as np
import pandas as pd
import mne
import scipy.stats
import matplotlib.pyplot as plt
import scipy

os.chdir('/isilon/LFMI/VMdrive/Lua/MEG_alpha_1-f_SCP_PLoSBiol_2024/scripts')
data_dir = '/isilon/LFMI/VMdrive/Lua/MEG_alpha_1-f_SCP_PLoSBiol_2024/data/HL/'
figures_dir = '/isilon/LFMI/VMdrive/Lua/MEG_alpha_1-f_SCP_PLoSBiol_2024/figures/'
supp_dir = '/isilon/LFMI/VMdrive/Lua/MEG_alpha_1-f_SCP_PLoSBiol_2024/scripts/supporting_files_toolboxes/HL/'

import HLTP
import pickle

#Import behavioral data in order to separate trials by recognized/unrecognized
all_df = pd.read_pickle(data_dir +'all_bhv_df.pkl')  

#Read in sensor names
with open(supp_dir + 'CTF275labels.txt') as f:
    sensor_names = f.readlines()
sensor_names = np.array(sensor_names)
sensor_names = np.delete(sensor_names,HLTP.bad_ch)

#Read in neighboring channels
neighbors = scipy.io.loadmat(supp_dir + 'ctf275_neighb.mat',simplify_cells=True)['neighbours']
neighbors = np.delete(neighbors,HLTP.bad_ch)

#Occipital channels
occipital = np.char.find(sensor_names,'O')
occipital_sensors = np.where(occipital==2)

# Sensors for each significant cluster
matalpha = scipy.io.loadmat(data_dir + 'alpha_cluster_sensors.mat')
matbeta = scipy.io.loadmat(data_dir + 'beta_cluster_sensors.mat')
alpha_clusters = matalpha['alpha_cluster_sensors'][0,:] - 1 #Subtracting 1 to account for Matlab indexing starting from 1 instead of 0
beta_clusters = matbeta['beta_cluster_sensors'][0,:] - 1 #Subtracting 1 to account for Matlab indexing starting from 1 instead of 0

with open(data_dir + 'DecisionVariables.pkl', 'rb') as f:
    DVs = pickle.load(f)
    
with open(data_dir + 'trial_indices_decisionvariables.pkl', 'rb') as f:
    trialIDs_dv = pickle.load(f)

def paired_standard_error(data_seen, data_unseen):
    if data_seen.shape != data_unseen.shape:
        raise ValueError("Data arrays must have the same shape")
    differences = data_seen - data_unseen
    # Ensure differences is always an array
    differences = np.atleast_1d(differences)
    # Compute standard deviation of differences
    sd_differences = np.std(differences, ddof=1, axis=0)
    # Compute paired standard error of the mean
    n = len(differences)
    paired_sem = sd_differences / np.sqrt(n)
    return paired_sem

def SEM_within_subject(data: np.ndarray) -> np.ndarray:
    """
    Calculate the standard error of the mean within subjects.

    All rows with any NaN are silently dropped from the calculation.

    Adapted from Baria, A. T., Maniscalco, B., & He, B. J. (2017). Initial-state-dependent, robust,
    transient neural dynamics encode conscious visual perception. PLoS computational biology,
    13(11), e1005806.

    Args:
        data (np.ndarray): numSubjects by numConditions 2D array of data.

    Returns:
        np.ndarray: numConditions 1D array of SEM values.
    """
    cleanData = data[~np.isnan(data).any(axis=1), :]
    N, M = cleanData.shape
    normedData = cleanData - cleanData.mean(axis=1, keepdims=True) + np.mean(cleanData)
    varNormed = np.var(normedData, axis=0, ddof=1) * M / (M - 1)
    stdNormed = np.sqrt(varNormed)
    return stdNormed / np.sqrt(N)

#Plot timecourse of DVs (only pre-stim)
timecourse_DVs = np.zeros((len(HLTP.subjects),2,DVs[0].shape[2]-1)) #subs x (seen,unseen) x timepoints
for s, subject in enumerate(HLTP.subjects):
    DVs[s][DVs[s] == np.inf] = np.nan
    DVs[s][DVs[s] == -np.inf] = np.nan
    timecourse_DVs[s,0,:] = np.nanmean(DVs[s][:,1,:],0)[:-1]
    timecourse_DVs[s,1,:] = np.nanmean(DVs[s][:,0,:],0)[:-1]
mean_timecourse_DVs = np.nanmean(timecourse_DVs,0)[:,3:]
paired_sem = []
for i in range(len(timecourse_DVs[0,0,3:])):
    paired_sem.append(SEM_within_subject(timecourse_DVs[:, :, 3+i]))
paired_sem = np.array(paired_sem)[:,0]
times = np.round(np.arange(-1.7,1.5,0.1),1) # 100 ms intervals

#Save to csv
df = pd.DataFrame({"DV_rec":mean_timecourse_DVs[0,:], "DV_unrec": mean_timecourse_DVs[1,:], "paired_sem": paired_sem, "time (s)": times})
df.to_csv(data_dir + "Fig2D.csv",index=False)

#PLOT
plt.clf()
fig, ax = plt.subplots(1, 1, figsize = (4, 2))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.plot(times,mean_timecourse_DVs[1,:], label = 'rec',color='deeppink')
plt.plot(times,mean_timecourse_DVs[0,:], label = 'unrec',color='royalblue')
plt.fill_between(times,
                 mean_timecourse_DVs[1,:] + paired_sem,
                 mean_timecourse_DVs[1,:] - paired_sem,
                 color='deeppink', alpha=0.2)
plt.fill_between(times,
                 mean_timecourse_DVs[0,:] + paired_sem,
                 mean_timecourse_DVs[0,:] - paired_sem,
                 color='royalblue', alpha=0.2)
minY = np.min(np.concatenate((mean_timecourse_DVs[0,:],mean_timecourse_DVs[1,:])))
timesAll = np.append(times,0)
plt.xticks(times[::4], rotation=-60, fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('time (s)',fontsize = 13)
plt.ylabel('mean SCP d.v.',fontsize = 13)
plt.axvline(0, color='red', linewidth=2)
plt.axhline(0, color='grey', linestyle='dashed', linewidth=2)
for x in [-1.7,-1.3,-0.9,-0.5,-0.1]:
    plt.axvline(x,color='grey',ls='dotted')
plt.xlim([-1.7, 1.5])
plt.ylim([-0.2, 0.2])
mag = ax.yaxis.get_offset_text()
mag.set_style('italic')
plt.legend(loc='upper right',frameon=False,bbox_to_anchor=[1.1, 1.32],labelspacing = 0.1,fontsize=12)
plt.savefig(figures_dir + "Fig2D.png",transparent=True,bbox_inches='tight', dpi=150)

#Plot timecourses of power 
all_sensors = [np.transpose(np.array(occipital_sensors)).astype(dtype=np.uint16),alpha_clusters[0],alpha_clusters[1],beta_clusters[0],beta_clusters[1]]
timecourse_alpha = np.load(data_dir + 'timecourse_alpha_allclusters.npy')[:,:,:,1:]
timecourse_beta = np.load(data_dir + 'timecourse_beta_allclusters.npy')[:,:,:,1:]

mean_alpha_time = np.nanmean(timecourse_alpha,axis=0) # (seen/unseen) x timepoints x clusters (occipital, alpha c1, alpha c2, beta c1, beta c2)
mean_beta_time = np.nanmean(timecourse_beta,axis=0)
errors_alpha = np.std(timecourse_alpha,axis=0) / np.sqrt(len(HLTP.subjects))
errors_beta = np.std(timecourse_beta,axis=0) / np.sqrt(len(HLTP.subjects))

times = np.round(np.arange(-1.7,1.8,0.1),1)[0::4] # 400 ms intervals
times_of_significance = np.array([1,3,2,3]) #time intervals at which each cluster was significant

plot_combinations = [(0, 0), (0, 1), (1, 2), (1, 3)]
savenames = ['Fig2E_1_timecourse','Fig2E_2_timecourse','Fig3C_1_timecourse','Fig3C_2_timecourse']

# Save to csv
df1 = pd.DataFrame({"alpha_seen":mean_alpha_time[0,:,0], "alpha_unseen": mean_alpha_time[1,:,0], "time (s)": times})
df1.to_csv(data_dir + savenames[0] + ".csv",index=False)

df2 = pd.DataFrame({"beta_seen":mean_alpha_time[0,:,1], "beta_unseen": mean_alpha_time[1,:,1], "time (s)": times})
df2.to_csv(data_dir + savenames[1] + ".csv",index=False)

df3 = pd.DataFrame({"beta_seen":mean_beta_time[0,:,2], "beta_unseen": mean_beta_time[1,:,2], "time (s)": times})
df3.to_csv(data_dir + savenames[2] + ".csv",index=False)

df4 = pd.DataFrame({"beta_seen":mean_beta_time[0,:,3], "beta_unseen": mean_beta_time[1,:,3], "time (s)": times})
df4.to_csv(data_dir + savenames[3] + ".csv",index=False)

for a, k in plot_combinations:
    plt.clf()
    fig, ax = plt.subplots(1, 1, figsize = (4, 2))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if a==0:
        plt.plot(times,mean_alpha_time[0,:,k], label = 'rec',color='deeppink')
        plt.plot(times,mean_alpha_time[1,:,k], label = 'unrec',color='royalblue')
        minY = np.min(np.concatenate((mean_alpha_time[0,:,k],mean_alpha_time[1,:,k])))
        maxY = np.max(np.concatenate((mean_alpha_time[0,:,k],mean_alpha_time[1,:,k])))
        plt.ylabel('alpha power (T$^2$)',fontsize = 13)
    else:
        plt.plot(times,mean_beta_time[0,:,k], label = 'rec',color='deeppink')
        plt.plot(times,mean_beta_time[1,:,k], label = 'unrec',color='royalblue')
        minY = np.min(np.concatenate((mean_beta_time[0,:,k],mean_beta_time[1,:,k])))
        maxY = np.max(np.concatenate((mean_beta_time[0,:,k],mean_beta_time[1,:,k])))
        plt.ylabel('beta power (T$^2$)',fontsize = 13)
    plt.xticks(times, rotation=-60,fontsize=12)
    plt.ylim([minY-0.2*(maxY - minY),maxY+0.2*(maxY - minY)])
    plt.yticks(fontsize=12)
    plt.xlabel('time (s)',fontsize = 13)
    plt.axvline(0, color='red', linewidth=2)
    for x in [-1.7,-1.3,-0.9,-0.5,-0.1]:
        plt.axvline(x,color='grey',ls='dotted')
    mag = ax.yaxis.get_offset_text()
    mag.set_style('italic')
    plt.legend(loc='upper right',frameon=False,bbox_to_anchor=[0.95, 1.1],labelspacing = 0.2,fontsize=12)
    plt.axvspan(times[int(times_of_significance[k])],times[int(times_of_significance[k])+1],alpha=0.2,color='grey')  
    plt.savefig(figures_dir + savenames[k] + ".png",transparent=True,bbox_inches='tight', dpi=150)
