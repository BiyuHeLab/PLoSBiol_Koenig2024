#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 16:44:51 2022

@author: koenil04
"""

import os
import numpy as np
import pandas as pd
import mne
import math
import pickle
import scipy.io
from scipy.stats import spearmanr, ttest_1samp, ttest_rel, norm, ranksums, wilcoxon 
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

os.chdir('/isilon/LFMI/VMdrive/Lua/MEG_alpha_1-f_SCP_PLoSBiol_2024/scripts')
data_dir = '/isilon/LFMI/VMdrive/Lua/MEG_alpha_1-f_SCP_PLoSBiol_2024/data/LL/'
figures_dir = '/isilon/LFMI/VMdrive/Lua/MEG_alpha_1-f_SCP_PLoSBiol_2024/figures/'
supp_dir = '/isilon/LFMI/VMdrive/Lua/MEG_alpha_1-f_SCP_PLoSBiol_2024/scripts/supporting_files_toolboxes/LL/'

DVs = scipy.io.loadmat(data_dir + 'DecisionVariables.mat')
DVs = DVs['DVs_time'][:,3:20] #subject x timepoints (take 3:20 indices to include -1.7s to -0.1s)
power_all = scipy.io.loadmat(data_dir + 'allpostfooofpower.mat')['Posc_all_dict']

nSubs = 11
n_prestim_timepoints = 4
nSensors = 273

#Obtain DVs for four timewindows of interest
DVs_sub = []
for sub in range(nSubs):
    DVs_onesub = np.zeros((DVs[sub,0].shape[0],n_prestim_timepoints))
    for time in range(n_prestim_timepoints):
            DVs_onesub[:,time] = np.mean(DVs[sub,(4*time):(4*time+4)])[:,0]
    DVs_sub.append(DVs_onesub)

matalpha = scipy.io.loadmat(data_dir + 'alpha_cluster_sensors.mat')
matbeta = scipy.io.loadmat(data_dir + 'beta_cluster_sensors.mat')
alpha_clusters = matalpha['alpha_cluster_sensors'][0,:] - 1 #Subtracting 1 to account for Matlab indexing starting from 1 instead of 0
beta_clusters = matbeta['beta_cluster_sensors'][0,:] - 1 #Subtracting 1 to account for Matlab indexing starting from 1 instead of 0

all_clusters = np.concatenate((alpha_clusters,beta_clusters))
cluster_times_alpha = np.array([0,1])
cluster_times_beta = np.array([0,0,1])
cluster_times = np.concatenate([cluster_times_alpha,cluster_times_beta]) #time intervals at which each cluster was significant

#Read in sensor names
with open(supp_dir + 'CTF275labels.txt') as f:
    sensor_names = f.readlines()
sensor_names = np.array(sensor_names)
sensor_names = np.delete(sensor_names,[172,191])

# Get all data into the same pandas dataframe
alldata_allsubs = pd.DataFrame(dtype=float)
for s in range(nSubs):
    for cluster in range(len(all_clusters)):
        #Import data for that subject and cluster
        dv = DVs_sub[s][:,cluster_times[cluster]]
        alpha_cluster_seen = np.nanmean(power_all[0,s*2][0,:,cluster_times[cluster],all_clusters[cluster]],(0,1)) #average alpha across sensors in that cluster at the time that cluster occurred and obtain alpha power for all seen trials
        alpha_cluster_unseen = np.nanmean(power_all[0,s*2+1][0,:,cluster_times[cluster],all_clusters[cluster]],(0,1)) #average alpha across sensors in that cluster at the time that cluster occurred and obtain alpha power for all unseen trials
        beta_cluster_seen = np.nanmean(power_all[0,s*2][4,:,cluster_times[cluster],all_clusters[cluster]],(0,1)) #average alpha across sensors in that cluster at the time that cluster occurred and obtain alpha power for all seen trials
        beta_cluster_unseen = np.nanmean(power_all[0,s*2+1][4,:,cluster_times[cluster],all_clusters[cluster]],(0,1)) #average alpha across sensors in that cluster at the time that cluster occurred and obtain alpha power for all unseen trials
        length_seen = alpha_cluster_seen.shape[0]
        length_unseen = alpha_cluster_unseen.shape[0]
        
        #Create pandas dataframe for that subject and cluster
        alldata = pd.DataFrame(
            {
                "subjID": s,
                "trialID": list(range(length_seen + length_unseen)),
                "clusterID": cluster,
                "alpha": np.nan,
                "beta": np.nan,
                "SCPdv": np.nan,
                "BehavRecognition": pd.Series(np.nan,index=range(length_seen + length_unseen),dtype='bool'),
                "BehavAccuracy": pd.Series(np.nan,index=range(length_seen + length_unseen),dtype='bool')
            })
        alldata['alpha'] = np.concatenate((alpha_cluster_seen,alpha_cluster_unseen))
        alldata['beta'] = np.concatenate((beta_cluster_seen,beta_cluster_unseen))
        alldata['SCPdv'] = dv
        alldata['BehavRecognition'] = np.repeat(np.array([1,0]),[length_seen,length_unseen]) #1 for seen, 0 for unseen
        alldata_allsubs = pd.concat([alldata_allsubs,alldata])
   
alldata_allsubs.replace([np.inf, -np.inf], np.nan, inplace=True)
alldata_allsubs.to_pickle(data_dir + "alldata_allsubjects_cluster.pkl")

#Convert alpha and beta values to femtoTeslas
alldata_allsubs['alpha'] = (np.sqrt(alldata_allsubs['alpha'])*10e-15)**2
alldata_allsubs['beta'] = (np.sqrt(alldata_allsubs['beta'])*10e-15)**2

# Fill out the correlation between SCPdv and alpha or beta power
cluster_corr_p = np.zeros((len(all_clusters),nSubs,2)) #all clusters by subjects by (alphaDV,betaDV)
cluster_corr_rho = np.zeros((len(all_clusters),nSubs,2))

for s in range(nSubs):
    for cluster in range(len(all_clusters)):
        df = alldata_allsubs[(alldata_allsubs.subjID == s) & (alldata_allsubs.clusterID == cluster)]
        df_clean = df.dropna()
        cluster_corr_rho[cluster,s,0], cluster_corr_p[cluster,s,0] = spearmanr(df_clean[['alpha']],df_clean[['SCPdv']])
        cluster_corr_rho[cluster,s,1], cluster_corr_p[cluster,s,1] = spearmanr(df_clean[['beta']],df_clean[['SCPdv']])

##Assessing significance of the correlations
cluster_corr_z = np.arctanh(cluster_corr_rho)

#Compute whether correlations are statistically significant across subjects
pvalues_clusters = np.zeros((2,all_clusters.shape[0])) #(alpha-DV, beta-DV) x all clusters
Wvalues_clusters = np.zeros((2,all_clusters.shape[0]))

for band in range(2):
    for cluster in range(len(all_clusters)):
        Wvalues_clusters[band,cluster], pvalues_clusters[band,cluster] = wilcoxon(cluster_corr_z[cluster,:,band])

#For Bayesian analysis in Jasp
np.savetxt(data_dir + 'alpha_SCP_corr_c1.csv', cluster_corr_z[0,:,0], delimiter=',', header='Value', comments='')
np.savetxt(data_dir + 'beta_SCP_corr_c1.csv', cluster_corr_z[1,:,1], delimiter=',', header='Value', comments='')
np.savetxt(data_dir + 'beta_SCP_corr_c2.csv', cluster_corr_z[2,:,1], delimiter=',', header='Value', comments='')

# Alpha vs. Beta correlations

cluster_corr_alphabeta_p = np.zeros((2,nSubs)) #alpha1 with beta 1, alpha 1 with beta 2
cluster_corr_alphabeta_rho = np.zeros((2,nSubs))

for s in range(nSubs):
    df_a1 = alldata_allsubs[(alldata_allsubs.subjID == s) & (alldata_allsubs.clusterID == 0)]
    df_b1 = alldata_allsubs[(alldata_allsubs.subjID == s) & (alldata_allsubs.clusterID == 1)]
    df_b2 = alldata_allsubs[(alldata_allsubs.subjID == s) & (alldata_allsubs.clusterID == 2)]
    
    cluster_corr_alphabeta_rho[0,s], cluster_corr_alphabeta_p[0,s] = spearmanr(df_a1[['alpha']],df_b1[['beta']],nan_policy='omit')
    cluster_corr_alphabeta_rho[1,s], cluster_corr_alphabeta_p[1,s] = spearmanr(df_a1[['alpha']],df_b2[['beta']],nan_policy='omit')
    
corr_alphabeta_z = np.arctanh(cluster_corr_alphabeta_rho)

pvalues_alphabeta = np.zeros((2))
Wvalues_alphabeta = np.zeros((2))
for corrnum in range(2):
    Wvalues_alphabeta[corrnum],pvalues_alphabeta[corrnum] = wilcoxon(corr_alphabeta_z[corrnum,:])

np.savetxt(data_dir + 'alpha_betac1_corr.csv', corr_alphabeta_z[0,:], delimiter=',', header='Value', comments='')
np.savetxt(data_dir + 'alpha_betac2_corr.csv', corr_alphabeta_z[1,:], delimiter=',', header='Value', comments='')
