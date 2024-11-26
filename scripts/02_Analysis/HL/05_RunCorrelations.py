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
from scipy.stats import spearmanr, ttest_1samp, ttest_rel, norm, ranksums, wilcoxon, ttest_ind
from scipy.stats.mstats import winsorize
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
box_style=dict(boxstyle='round', facecolor='wheat', alpha=0.5)

os.chdir('/isilon/LFMI/VMdrive/Lua/MEG_alpha_1-f_SCP_PLoSBiol_2024/scripts')
data_dir = '/isilon/LFMI/VMdrive/Lua/MEG_alpha_1-f_SCP_PLoSBiol_2024/data/HL/'
figures_dir = '/isilon/LFMI/VMdrive/Lua/MEG_alpha_1-f_SCP_PLoSBiol_2024/figures/'
supp_dir = '/isilon/LFMI/VMdrive/Lua/MEG_alpha_1-f_SCP_PLoSBiol_2024/scripts/supporting_files_toolboxes/HL/'

import HLTP

DVs = np.load(data_dir + 'DecisionVariables_logit.pkl',allow_pickle=True) #Format is list of nSubs, for each trials x (seen,unseen probability) x time
trialIDs_dv = np.load(data_dir + 'trial_indices_decisionvariables.pkl',allow_pickle=True)
alpha_real = np.load(data_dir + 'alpha_alltrials.npy',allow_pickle=True)
beta_real = np.load(data_dir + 'beta_alltrials.npy',allow_pickle=True) 
trialIDs_alpha_real = np.load(data_dir + 'trialIDs_real.npy',allow_pickle=True) #these are the same for beta also

#Subject 16 has outlier data identified in the alpha power data that needs to be removed from DV
remove_seen_s16 = np.array([258]) #these are absolute indices, 
remove_unseen_s16 = np.array([123,192,226,240,248,252,266,313])
remove_s16 = np.concatenate([remove_seen_s16,remove_unseen_s16])
trialIDs_dv[16] = np.delete(trialIDs_dv[16],np.where(np.isin(trialIDs_dv[16],remove_s16)))

alpha_scr = []
alpha_scr_allsensors = []
beta_scr = []
for s, subject in enumerate(HLTP.subjects):
    alpha_sub = np.load(data_dir + 'Preprocessed/' + subject + '/AUC_power_postfooof_scrambled.npy')[0,:,:,:]
    alpha_scr.append(alpha_sub)
    beta_sub = np.load(data_dir + 'Preprocessed/' + subject + '/AUC_power_postfooof_scrambled.npy')[1,:,:,:]
    beta_scr.append(beta_sub)
alpha_scr = np.array(alpha_scr, dtype=object)
beta_scr = np.array(beta_scr, dtype=object)
trialIDs_alpha_scr = np.load(data_dir + 'trialIDs_scrambled.npy',allow_pickle=True) #These are the same for beta also

#Get trial IDs for all alpha data (real + scrambled)
trialIDs_all = []
for s, _ in enumerate(HLTP.subjects):
    trialIDs_all.append(np.concatenate([trialIDs_alpha_real[s],trialIDs_alpha_scr[s]]))

matalpha = scipy.io.loadmat(data_dir + 'alpha_cluster_sensors.mat')
matbeta = scipy.io.loadmat(data_dir + 'beta_cluster_sensors.mat')
alpha_clusters = matalpha['alpha_cluster_sensors'][0,:] - 1 #Subtracting 1 to account for Matlab indexing starting from 1 instead of 0
beta_clusters = matbeta['beta_cluster_sensors'][0,:] - 1 #Subtracting 1 to account for Matlab indexing starting from 1 instead of 0

all_clusters = np.concatenate((alpha_clusters,beta_clusters))
cluster_times_alpha = np.array([1,3])
cluster_times_beta = np.array([2,3])
cluster_times = np.concatenate([cluster_times_alpha,cluster_times_beta]) #time intervals at which each cluster was significant

#Read in sensor names
with open(supp_dir + 'CTF275labels.txt') as f:
    sensor_names = f.readlines()
sensor_names = np.array(sensor_names)
sensor_names = np.delete(sensor_names,HLTP.bad_ch)

#Occipital channels
occipital = np.char.find(sensor_names,'O')
occipital_sensors = np.where(occipital==2)[0]

#import behav data for criterion computation
all_df = pd.read_pickle(data_dir +'all_bhv_df.pkl')  

#Import pupil data
Pupil_dict = {}
Pupil_dict['pupil_size'] = {}
Pupil_dict['trial'] = {}
Pupil_dict['bad_trials'] = {}
excluded = 'NC' # no pupil data due to bad quality eye-tracking. 
for sub_idx, sub in enumerate(HLTP.subjects):
    if sub == excluded: continue
    Pupil_dict['pupil_size'][sub] = np.load(
            HLTP.MEG_pro_dir + '/' + sub + '/prestim_pupil_nointerp_av.npy')
    Pupil_dict['trial'][sub] = np.arange(len(Pupil_dict['pupil_size'][sub]))
    
#Function for computing SDT measures
def find_sdt(behavdat): #Load in behavior for one subject dataframe with BehavRealScr and BehavRecognition as columns
    proportion_R = behavdat.groupby('BehavRealScr').mean().to_numpy()
    count_R = behavdat.groupby('BehavRealScr').count().to_numpy()
    hitRate = proportion_R[1][0]
    n_real_img = count_R[1][0]
    faRate =  proportion_R[0][0]
    n_scram_img =  count_R[0][0]
    # Correct the values of 0 and 1:
    faRate = max( faRate, 1. / (2 * n_scram_img))
    hitRate = min( hitRate, 1 - (1. / (2 * n_real_img)))
    Z = scipy.stats.norm.ppf
    d_prime = Z(hitRate)- Z(faRate)
    c = -(Z(hitRate) + Z(faRate))/2
    return(c, d_prime, hitRate)

# Get all data into the same pandas dataframe
alldata_allsubs = pd.DataFrame(dtype=float)
for s, sub in enumerate(HLTP.subjects):
    for cluster in range(len(all_clusters)):
        #Import data for that subject and cluster
        dv = DVs[s][:,1,cluster_times[cluster]]
        alpha_cluster_real = np.nanmean(alpha_real[s][:,all_clusters[cluster],cluster_times[cluster]],axis=1)[:,0]
        alpha_cluster_scr = np.nanmean(alpha_scr[s][:,all_clusters[cluster],cluster_times[cluster]],axis=1)[:,0]
        alpha_in_cluster_all = np.concatenate((alpha_cluster_real,alpha_cluster_scr))
        beta_cluster_real = np.nanmean(beta_real[s][:,all_clusters[cluster],cluster_times[cluster]],axis=1)[:,0]
        beta_cluster_scr = np.nanmean(beta_scr[s][:,all_clusters[cluster],cluster_times[cluster]],axis=1)[:,0]
        beta_in_cluster_all = np.concatenate((beta_cluster_real,beta_cluster_scr))
        behav = all_df[all_df.subject == sub][['real_img','seen','correct']]
        
        #Create pandas dataframe for that subject and cluster
        alldata = pd.DataFrame(
            {
                "subjID": sub,
                "trialID": list(range(360)),
                "clusterID": cluster,
                "alpha": np.nan,
                "beta": np.nan,
                "SCPdv": np.nan,
                "PupilSize": np.nan,
                "BehavRealScr": pd.Series(np.nan,index=range(360),dtype='bool'),
                "BehavRecognition": pd.Series(np.nan,index=range(360),dtype='bool'),
                "BehavAccuracy": pd.Series(np.nan,index=range(360),dtype='bool')
            })
        alldata.loc[trialIDs_all[s],'alpha'] = alpha_in_cluster_all
        alldata.loc[trialIDs_all[s],'beta'] = beta_in_cluster_all
        alldata.loc[trialIDs_dv[s],'SCPdv'] = dv
        if sub != 'NC':
            alldata.loc[Pupil_dict['trial'][sub],'PupilSize'] = Pupil_dict['pupil_size'][sub]
        alldata.loc[behav.index,'BehavRealScr'] = behav.real_img
        alldata.loc[behav.index,'BehavRecognition'] = behav.seen
        alldata.loc[behav.index,'BehavAccuracy'] = behav.correct
        alldata_allsubs = pd.concat([alldata_allsubs,alldata])
        
alldata_allsubs.replace([np.inf, -np.inf], np.nan, inplace=True)
alldata_allsensors.replace([np.inf, -np.inf], np.nan, inplace=True)

alldata_allsubs.to_pickle(data_dir + alldata_allsubjects_cluster_withpupil.pkl")

alldata_allsubs = pd.read_pickle(data_dir + "alldata_allsubjects_cluster_withpupil.pkl")
allSCP_pupildata = pd.read_pickle(data_dir + "allpupil_SCP_data.pkl")
from scipy.special import logit
allSCP_pupildata['pred_prob'] = logit(allSCP_pupildata['pred_prob'])
allSCP_pupildata.replace([np.inf, -np.inf], np.nan, inplace=True)

# Fill out the correlation between SCPdv and alpha or beta power
cluster_corr_p = np.zeros((len(all_clusters),len(HLTP.subjects),2)) #all clusters by subjects by (alphaDV,betaDV)
cluster_corr_rho = np.zeros((len(all_clusters),len(HLTP.subjects),2))

for s, sub in enumerate(HLTP.subjects):
    for cluster in range(len(all_clusters)):
        df = alldata_allsubs[(alldata_allsubs.subjID == sub) & (alldata_allsubs.clusterID == cluster)]
        df_clean = df.dropna() #Careful, because df now contains pupilsize column, and there is no data for s = 16, it removes s = 16 from all further analyses/plotting even if they don't involve pupil size. Stats and plots were made before this change
        
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
        
corr_SCP_pupil_z = np.load(data_dir + 'SCP_pupil_correlation.npy')
corr_SCP_pupil_z = np.delete(corr_SCP_pupil_z,16) #Sub 16 has no pupil data
Wvalue_SCP_pupil, pvalue_SCP_pupil = wilcoxon(corr_SCP_pupil_z)

## AUROC ANALYSES ###

def compute_auroc(labels, data):
    thresholds = np.sort(data)
    fpr = np.zeros(len(thresholds))
    tpr = np.zeros(len(thresholds))
    for i, thr in enumerate(thresholds):
        model_outcome = data >= thr
        tpr[i] = np.sum((model_outcome == True) & (labels == True)) / (np.sum((model_outcome == True) & (labels == True)) + np.sum((model_outcome == False) & (labels == True)))
        fpr[i] = np.sum((model_outcome == True) & (labels == False)) / (np.sum((model_outcome == True) & (labels == False)) + np.sum((model_outcome == False) & (labels == False)))
    fpr = np.flip(fpr)
    tpr = np.flip(tpr)
    auroc = 1 - np.trapz(fpr,tpr)
    return(auroc)

def obtain_residuals(x,y): # x = inputs (regressors) that we want to regress out; y = responses that we want to model and want to obtain residuals for
    x = np.array(x).reshape((-1,1)) # reshape to obtain a two-dimensional array that is one column long and as many rows as data inputs
    y = np.array(y)
    model = LinearRegression().fit(x,y)
    y_pred = model.coef_ * x[:,0] + model.intercept_
    residuals = y - y_pred
    return(residuals)

alpha_auroc = np.zeros((len(HLTP.subjects),len(all_clusters)))
alpha_nobeta_auroc = np.zeros((len(HLTP.subjects),len(all_clusters)))

beta_auroc = np.zeros((len(HLTP.subjects),len(all_clusters)))
beta_noSCP_auroc = np.zeros((len(HLTP.subjects),len(all_clusters)))
beta_noalpha_auroc = np.zeros((len(HLTP.subjects),len(all_clusters)))

SCP_auroc = np.zeros((len(HLTP.subjects),len(all_clusters)))
SCP_nobeta_auroc = np.zeros((len(HLTP.subjects),len(all_clusters)))

for s, sub in enumerate(HLTP.subjects):
    for cluster in range(len(all_clusters)):
        df = alldata_allsubs[(alldata_allsubs.subjID == sub) & (alldata_allsubs.clusterID == cluster)]
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_clean = df.dropna()
        
        df_clean["constant"] = 1
        df_clean['alpha_res_nobeta'] = sm.OLS(df_clean['alpha'],df_clean['beta']).fit().resid
        df_clean['beta_res_noSCP'] = sm.OLS(df_clean['beta'],df_clean['SCPdv']).fit().resid
        df_clean['beta_res_noalpha'] = sm.OLS(df_clean['beta'],df_clean['alpha']).fit().resid
        df_clean['SCPdv_res_nobeta'] = sm.OLS(df_clean['SCPdv'],df_clean['beta']).fit().resid
        
        #Normalize all columns of interest and round to 3rd decimal
        df_clean[['alpha_norm','beta_norm','SCPdv_norm','alpha_res_nobeta_norm','beta_res_noSCP_norm','beta_res_noalpha_norm','SCPdv_res_nobeta_norm']] = (df_clean[['alpha','beta','SCPdv','alpha_res_nobeta','beta_res_noSCP','beta_res_noalpha','SCPdv_res_nobeta']] - df_clean[['alpha','beta','SCPdv','alpha_res_nobeta','beta_res_noSCP','beta_res_noalpha','SCPdv_res_nobeta']].min()) / (df_clean[['alpha','beta','SCPdv','alpha_res_nobeta','beta_res_noSCP','beta_res_noalpha','SCPdv_res_nobeta']].max() - df_clean[['alpha','beta','SCPdv','alpha_res_nobeta','beta_res_noSCP','beta_res_noalpha','SCPdv_res_nobeta']].min())
        df_clean[['alpha_norm','beta_norm','SCPdv_norm','alpha_res_nobeta_norm','beta_res_noSCP_norm','beta_res_noalpha_norm','SCPdv_res_nobeta_norm']] = df_clean[['alpha_norm','beta_norm','SCPdv_norm','alpha_res_nobeta_norm','beta_res_noSCP_norm','beta_res_noalpha_norm','SCPdv_res_nobeta_norm']].round(3)
        
        #Obtain AUROC scores
        alpha_auroc[s,cluster] = compute_auroc(df_clean['BehavRecognition'],df_clean['alpha_norm'])
        beta_auroc[s,cluster] = compute_auroc(df_clean['BehavRecognition'],df_clean['beta_norm'])
        SCP_auroc[s,cluster] = compute_auroc(df_clean['BehavRecognition'],df_clean['SCPdv_norm'])
        alpha_nobeta_auroc[s,cluster] = compute_auroc(df_clean['BehavRecognition'],df_clean['alpha_res_nobeta_norm'])
        beta_noSCP_auroc[s,cluster] = compute_auroc(df_clean['BehavRecognition'],df_clean['beta_res_noSCP_norm'])
        beta_noalpha_auroc[s,cluster] = compute_auroc(df_clean['BehavRecognition'],df_clean['beta_res_noalpha_norm'])
        SCP_nobeta_auroc[s,cluster] = compute_auroc(df_clean['BehavRecognition'],df_clean['SCPdv_res_nobeta_norm'])

np.save(data_dir + 'alpha_auroc.npy',alpha_auroc)
np.save(data_dir + 'beta_auroc.npy',beta_auroc)
np.save(data_dir + 'SCP_auroc',SCP_auroc)
np.save(data_dir + 'alpha_nobeta_auroc.npy',alpha_nobeta_auroc)
np.save(data_dir + 'beta_noSCP_auroc.npy',beta_noSCP_auroc)
np.save(data_dir + 'beta_noalpha_auroc.npy',beta_noalpha_auroc)
np.save(data_dir + 'SCP_nobeta_auroc.npy',SCP_nobeta_auroc)

### MEDIATION ANALYSIS

## TESTING MEDIATION OF SCP ON RECOGNITION VIA PUPIL
## For this one don't use clusters -- use estimates across entire pre-stimulus time window.    
#GLM1: pupil (M) = i + a * SCP (X) + e
#GLM2: recognition (Y) = i + b * pupil (M) + e
#GLM3: recognition (Y) = i + c * SCP (X) + e
#GLM4: recognition (Y) = i + c' * SCP (X) + b' * pupil (M) + e
a_all = np.zeros((len(HLTP.subjects)))
b_all = np.zeros((len(HLTP.subjects)))
c_all = np.zeros((len(HLTP.subjects)))
bprime_all = np.zeros((len(HLTP.subjects)))
cprime_all = np.zeros((len(HLTP.subjects)))

from sklearn.linear_model import LinearRegression

behavdat = pd.DataFrame(dtype=float)

for s, sub in enumerate(HLTP.subjects):
    indices = allSCP_pupildata[(allSCP_pupildata.subjID == sub)].index
    df = alldata_allsubs[(alldata_allsubs.subjID == sub) & (alldata_allsubs.clusterID == 0)]
    behavdat = pd.concat([behavdat,df.loc[indices,'BehavRecognition']])
    
allSCP_pupildata['BehavRecognition'] = behavdat

for s, sub in enumerate(HLTP.subjects):
    if s == 16:
        a_all[s] = np.nan
        b_all[s] = np.nan
        c_all[s] = np.nan
        bprime_all[s] = np.nan
        cprime_all[s] = np.nan
        continue
    df = allSCP_pupildata[(allSCP_pupildata.subjID == sub)]
    df_clean= df.dropna()
    
    model1 = LinearRegression().fit(df_clean.pred_prob.values.reshape(-1, 1),df_clean.pupil_size.values.reshape(-1,1))
    a_all[s] = model1.coef_
    
    model2 = LinearRegression().fit(df_clean.pupil_size.values.reshape(-1, 1),df_clean.BehavRecognition.values.reshape(-1,1))
    b_all[s] = model2.coef_
    
    model3 = LinearRegression().fit(df_clean.pred_prob.values.reshape(-1, 1),df_clean.BehavRecognition.values.reshape(-1,1))
    c_all[s] = model3.coef_
    
    model4 = LinearRegression().fit(df_clean[['pupil_size','pred_prob']].values,df_clean.BehavRecognition.values.reshape(-1,1))
    bprime_all[s], cprime_all[s] = model4.coef_[0]


## TESTING MEDIATION OF ALPHA ON RECOGNITION VIA PUPIL

#GLM1: pupil (M) = i + a * alpha (X) + e
#GLM2: recognition (Y) = i + b * pupil (M) + e
#GLM3: recognition (Y) = i + c * alpha (X) + e
#GLM4: recognition (Y) = i + c' * alpha (X) + b' * pupil (M) + e
a_all = np.zeros((len(HLTP.subjects),len(all_clusters)))
b_all = np.zeros((len(HLTP.subjects),len(all_clusters)))
c_all = np.zeros((len(HLTP.subjects),len(all_clusters)))
bprime_all = np.zeros((len(HLTP.subjects),len(all_clusters)))
cprime_all = np.zeros((len(HLTP.subjects),len(all_clusters)))

from sklearn.linear_model import LinearRegression

for s, sub in enumerate(HLTP.subjects):
    if s == 16:
        a_all[s,:] = np.nan
        b_all[s,:] = np.nan
        c_all[s,:] = np.nan
        bprime_all[s,:] = np.nan
        cprime_all[s,:] = np.nan
        continue
    for cluster in range(len(all_clusters)):
        df = alldata_allsubs[(alldata_allsubs.subjID == sub) & (alldata_allsubs.clusterID == cluster)]
        df_clean = df.dropna()
        
        model1 = LinearRegression().fit(df_clean.alpha.values.reshape(-1, 1),df_clean.PupilSize.values.reshape(-1,1))
        a_all[s,cluster] = model1.coef_
        
        model2 = LinearRegression().fit(df_clean.PupilSize.values.reshape(-1, 1),df_clean.BehavRecognition.values.reshape(-1,1))
        b_all[s,cluster] = model2.coef_
        
        model3 = LinearRegression().fit(df_clean.alpha.values.reshape(-1, 1),df_clean.BehavRecognition.values.reshape(-1,1))
        c_all[s,cluster] = model3.coef_
        
        model4 = LinearRegression().fit(df_clean[['PupilSize','alpha']].values,df_clean.BehavRecognition.values.reshape(-1,1))
        bprime_all[s,cluster], cprime_all[s,cluster] = model4.coef_[0]


## TESTING MEDIATION OF BETA ON RECOGNITION VIA PUPIL

#GLM1: pupil (M) = i + a * beta (X) + e
#GLM2: recognition (Y) = i + b * pupil (M) + e
#GLM3: recognition (Y) = i + c * beta (X) + e
#GLM4: recognition (Y) = i + c' * beta (X) + b' * pupil (M) + e
a_all = np.zeros((len(HLTP.subjects),len(all_clusters)))
b_all = np.zeros((len(HLTP.subjects),len(all_clusters)))
c_all = np.zeros((len(HLTP.subjects),len(all_clusters)))
bprime_all = np.zeros((len(HLTP.subjects),len(all_clusters)))
cprime_all = np.zeros((len(HLTP.subjects),len(all_clusters)))

from sklearn.linear_model import LinearRegression

for s, sub in enumerate(HLTP.subjects):
    if s == 16:
        a_all[s,:] = np.nan
        b_all[s,:] = np.nan
        c_all[s,:] = np.nan
        bprime_all[s,:] = np.nan
        cprime_all[s,:] = np.nan
        continue
    for cluster in range(len(all_clusters)):
        df = alldata_allsubs[(alldata_allsubs.subjID == sub) & (alldata_allsubs.clusterID == cluster)]
        df_clean = df.dropna()
        
        model1 = LinearRegression().fit(df_clean.beta.values.reshape(-1, 1),df_clean.PupilSize.values.reshape(-1,1))
        a_all[s,cluster] = model1.coef_
        
        model2 = LinearRegression().fit(df_clean.PupilSize.values.reshape(-1, 1),df_clean.BehavRecognition.values.reshape(-1,1))
        b_all[s,cluster] = model2.coef_
        
        model3 = LinearRegression().fit(df_clean.beta.values.reshape(-1, 1),df_clean.BehavRecognition.values.reshape(-1,1))
        c_all[s,cluster] = model3.coef_
        
        model4 = LinearRegression().fit(df_clean[['PupilSize','beta']].values,df_clean.BehavRecognition.values.reshape(-1,1))
        bprime_all[s,cluster], cprime_all[s,cluster] = model4.coef_[0]