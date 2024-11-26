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
data_dir = '/isilon/LFMI/VMdrive/Lua/MEG_alpha_1-f_SCP_PLoSBiol_2024/data/LL/'
figures_dir = '/isilon/LFMI/VMdrive/Lua/MEG_alpha_1-f_SCP_PLoSBiol_2024/figures/'
supp_dir = '/isilon/LFMI/VMdrive/Lua/MEG_alpha_1-f_SCP_PLoSBiol_2024/scripts/supporting_files_toolboxes/LL/'

nSubs=11

#Read in sensor names
with open(supp_dir + 'CTF275labels.txt') as f:
    sensor_names = f.readlines()
sensor_names = np.array(sensor_names)
sensor_names = np.delete(sensor_names,[172,191])
#Read in neighboring channels
neighbors = scipy.io.loadmat(supp_dir + 'ctf275_neighb.mat',simplify_cells=True)['neighbours']
neighbors = np.delete(neighbors,[172,191])

#Occipital channels
occipital = np.char.find(sensor_names,'O')
occipital_sensors = np.where(occipital==2)

# Sensors for each significant cluster
matalpha = scipy.io.loadmat(data_dir + 'alpha_cluster_sensors.mat')
matbeta = scipy.io.loadmat(data_dir + 'beta_cluster_sensors.mat')
alpha_clusters = matalpha['alpha_cluster_sensors'][0,:] - 1 #Subtracting 1 to account for Matlab indexing starting from 1 instead of 0
beta_clusters = matbeta['beta_cluster_sensors'][0,:] - 1 #Subtracting 1 to account for Matlab indexing starting from 1 instead of 0

#Import alpha and beta power
power_all = scipy.io.loadmat(data_dir + 'allpostfooofpower')['Posc_all_dict']

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

#Import DVs
DVs = scipy.io.loadmat(data_dir + 'DecisionVariables.mat')
DVs = DVs['DVs_time'][:,3:35] #subject x timepoints (take 3: indices to include -1.7s to ...)
cond_n = scipy.io.loadmat(data_dir + 'trial_numbers_2conds.mat')['n'][0,:]
t_n = DVs.shape[1]
nSubs = len(DVs)
DV_time = np.zeros((nSubs, t_n, 2))  # timepoints by conditions (seen/unseen)
for sub in range(nSubs):
    for time in range(t_n):
        DV_time[sub, time, 0] = np.mean(DVs[sub,time][0:int(cond_n[sub][0][0])])
        DV_time[sub, time, 1] = np.mean(DVs[sub,time][int(cond_n[sub][0][0]):])

DVs_mean = scipy.io.loadmat(data_dir + 'DV_time_avg.mat')
DVs_mean = DVs_mean['DV_time_avg']
DVs_mean = DVs_mean[0:32,:]
times = np.round(np.arange(-1.7,1.5,0.1),1) # 400 ms intervals
paired_sem = []
for i in range(len(DV_time[0,:,0])):
    paired_sem.append(SEM_within_subject(DV_time[:,i,:]))
paired_sem = np.array(paired_sem)[:,0]

#PLOT
plt.clf()
fig, ax = plt.subplots(1, 1, figsize = (4, 2))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.plot(times,DVs_mean[:,0], label = 'seen',color='deeppink')
plt.plot(times,DVs_mean[:,1], label = 'unseen',color='royalblue')
plt.fill_between(times,
                 DVs_mean[:,0] + paired_sem,
                 DVs_mean[:,0] - paired_sem,
                 color='deeppink', alpha=0.2)
plt.fill_between(times,
                 DVs_mean[:,1] + paired_sem,
                 DVs_mean[:,1] - paired_sem,
                 color='royalblue', alpha=0.2)
minY = np.min(np.concatenate((DVs_mean[0,:],DVs_mean[1,:])))
plt.xticks(times[0::4], rotation=-60,fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('time (s)',fontsize = 13)
plt.ylabel('mean SCP d.v.',fontsize = 13)
plt.axvline(0, color='red', linewidth=2)
plt.axhline(0, color='grey', linestyle='dashed', linewidth=2)
for x in [-1.7,-1.3,-0.9,-0.5,-0.1]:
    plt.axvline(x,color='grey',ls='dotted')
mag = ax.yaxis.get_offset_text()
mag.set_style('italic')
plt.xlim([-1.7,1.5])
#plt.suptitle("Timecourse in " + names[k],y=1.02)
plt.legend(loc='upper right',frameon=False,bbox_to_anchor=[1.1, 1.1],labelspacing = 0.1,fontsize=12)
plt.savefig("../figures/Fig2A.png",transparent=True,bbox_inches='tight', dpi=150)

#Plot timecourses of power 
timecourse_alpha = np.zeros((nSubs,2,9,3)) #subjects x (seen,unseen) x timepoints x (sensors of alpha cluster #1, sensors of beta clusters #1, #2)
timecourse_beta = np.zeros((nSubs,2,9,3)) #subjects  x (seen,unseen) x timepoints x (sensors of alpha cluster #1, sensors of beta clusters #1, #2)

all_sensors = [alpha_clusters[0],beta_clusters[0],beta_clusters[1]]

for s in range(nSubs):
    for x, sensors_of_interest in enumerate(all_sensors):
        timecourse_alpha[s,0,:,x] = np.nanmean(power_all[0,s*2][0,:,:,sensors_of_interest],(0,1,2)) #seen
        timecourse_alpha[s,1,:,x] = np.nanmean(power_all[0,s*2+1][0,:,:,sensors_of_interest],(0,1,2)) #unseen
        
        timecourse_beta[s,0,:,x] = np.nanmean(power_all[0,s*2][3,:,:,sensors_of_interest],(0,1,2)) #seen
        timecourse_beta[s,1,:,x] = np.nanmean(power_all[0,s*2+1][3,:,:,sensors_of_interest],(0,1,2)) #unseen

#Convert to femtoTesla to match other dataset
timecourse_alpha = (np.sqrt(timecourse_alpha)*10e-15)**2
timecourse_beta = (np.sqrt(timecourse_beta)*10e-15)**2

mean_alpha_time = np.nanmean(timecourse_alpha,axis=0) # (seen/unseen) x timepoints x clusters (occipital, alpha c1, alpha c2, beta c1, beta c2)
mean_beta_time = np.nanmean(timecourse_beta,axis=0)

errors_alpha = np.std(timecourse_alpha,axis=0) / np.sqrt(nSubs)
errors_beta = np.std(timecourse_beta,axis=0) / np.sqrt(nSubs)

names = ['alpha cluster # 1','beta cluster #1', 'beta cluster # 2']
savenames = ['Fig2B_timecourse','Fig3A_1_timecourse','Fig3A_2_timecourse']
times = np.round(np.arange(-1.7,1.8,0.1),1)[0::4] # 400 ms intervals
times_of_significance = np.array([0,0,0]) #time intervals at which each cluster was significant

for k, sensors_of_interest in enumerate(all_sensors):
    plt.clf()
    fig, ax = plt.subplots(1, 1, figsize = (4, 2))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if k==0:
        plt.plot(times,mean_alpha_time[0,:,k], label = 'seen',color='deeppink')
        plt.plot(times,mean_alpha_time[1,:,k], label = 'unseen',color='royalblue')
        minY = np.min(np.concatenate((mean_alpha_time[0,:,k],mean_alpha_time[1,:,k])))
        maxY = np.max(np.concatenate((mean_alpha_time[0,:,k],mean_alpha_time[1,:,k])))
        plt.ylabel('alpha power (T$^2$)', fontsize=13)
    else:
        plt.plot(times,mean_beta_time[0,:,k], label = 'seen',color='deeppink')
        plt.plot(times,mean_beta_time[1,:,k], label = 'unseen',color='royalblue')
        minY = np.min(np.concatenate((mean_beta_time[0,:,k],mean_beta_time[1,:,k])))
        maxY = np.max(np.concatenate((mean_beta_time[0,:,k],mean_beta_time[1,:,k])))
        plt.ylabel('beta power (T$^2$)',fontsize = 13)
    plt.xticks(times, rotation=-60,fontsize=12)
    plt.ylim([minY-0.2*(maxY - minY),maxY+0.2*(maxY - minY)])
    plt.yticks(fontsize=12)
    plt.xlabel('time (s)',fontsize=13)
    plt.axvline(0, color='red',linewidth = 2)
    for x in [-1.7,-1.3,-0.9,-0.5,-0.1]:
        plt.axvline(x,color='grey',ls='dotted')
    plt.legend(loc='upper right',frameon=False,bbox_to_anchor=[0.95, 1.0],labelspacing = 0.2,fontsize=12)
    mag = ax.yaxis.get_offset_text()
    mag.set_style('italic')
    plt.axvspan(times[int(times_of_significance[k])],times[int(times_of_significance[k])+1],alpha=0.2,color='grey')  
    plt.savefig(figures_dir + savenames[k] + ".png",transparent=True,bbox_inches='tight', dpi=150)