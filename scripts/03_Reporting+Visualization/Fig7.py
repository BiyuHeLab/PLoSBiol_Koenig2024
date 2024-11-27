#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 16:44:51 2022

@author: koenil04
"""

import os
import numpy as np
import pandas as pd
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

alldata_allsubs = pd.read_pickle(data_dir + "alldata_allsubjects_cluster_withpupil_andaperiodic.pkl")
allSCP_pupildata = pd.read_pickle(data_dir + "allpupil_SCP_data.pkl")
from scipy.special import logit
allSCP_pupildata['pred_prob'] = logit(allSCP_pupildata['pred_prob'])
allSCP_pupildata.replace([np.inf, -np.inf], np.nan, inplace=True)

matalpha = scipy.io.loadmat(data_dir + 'alpha_cluster_sensors.mat')
matbeta = scipy.io.loadmat(data_dir + 'beta_cluster_sensors.mat')
alpha_clusters = matalpha['alpha_cluster_sensors'][0,:] - 1 #Subtracting 1 to account for Matlab indexing starting from 1 instead of 0
beta_clusters = matbeta['beta_cluster_sensors'][0,:] - 1 #Subtracting 1 to account for Matlab indexing starting from 1 instead of 0

all_clusters = np.concatenate((alpha_clusters,beta_clusters))

### MEDIATION ANALYSIS

## TESTING MEDIATION OF ALPHA ON RECOGNITION VIA PUPIL AND PUPIL VIA ALPHA

#GLM1: pupil (M) = i + a * alpha (X) + e
#GLM2: recognition (Y) = i + b * pupil (M) + e
#GLM3: recognition (Y) = i + c * alpha (X) + e
#GLM4: recognition (Y) = i + c' * alpha (X) + b' * pupil (M) + e
a_all = np.zeros((len(HLTP.subjects),2))
b_all = np.zeros((len(HLTP.subjects),2))
c_all = np.zeros((len(HLTP.subjects),2))
d_all = np.zeros((len(HLTP.subjects),2))
e_all = np.zeros((len(HLTP.subjects),2))
f_all = np.zeros((len(HLTP.subjects),2))

r2_a = np.zeros((len(HLTP.subjects),2))
r2_b = np.zeros((len(HLTP.subjects),2))
auc_c = np.zeros((len(HLTP.subjects),2))
auc_d = np.zeros((len(HLTP.subjects),2))
auc_e = np.zeros((len(HLTP.subjects),2))
auc_f = np.zeros((len(HLTP.subjects),2))

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score
from scipy.stats import zscore
import scipy.stats as stats

for s, sub in enumerate(HLTP.subjects):
    if s == 16:
        a_all[s,:] = np.nan
        b_all[s,:] = np.nan
        c_all[s,:] = np.nan
        d_all[s,:] = np.nan
        e_all[s,:] = np.nan
        f_all[s,:] = np.nan
        r2_a[s] = np.nan
        r2_b[s] = np.nan
        auc_c[s] = np.nan
        auc_d[s] = np.nan
        auc_e[s] = np.nan
        auc_f[s] = np.nan
        continue
    for cluster in range(len(all_clusters[:2])):
        df = alldata_allsubs[(alldata_allsubs.subjID == sub) & (alldata_allsubs.clusterID == cluster)]
        df_clean = df.dropna()
        
        varA = df_clean.alpha.values.reshape(-1, 1)
        varB = df_clean.PupilSize.values.reshape(-1,1)
        varY = df_clean.BehavRecognition.values.reshape(-1,1)
        varAB = df_clean[['PupilSize','alpha']].values
        
        #Normalize z-score
        varA_norm = zscore(varA)
        varB_norm = zscore(varB)
        varAB_norm = np.hstack((varA_norm,varB_norm))
        
        # Direct models
        model1 = LinearRegression().fit(varA_norm, varB_norm)
        a_all[s, cluster] = model1.coef_
        
        model2 = LinearRegression().fit(varB_norm, varA_norm)
        b_all[s, cluster] = model2.coef_
        
        model3 = LogisticRegression().fit(varB_norm, varY)
        c_all[s, cluster] = model3.coef_
        
        model4 = LogisticRegression().fit(varA_norm, varY)
        d_all[s, cluster] = model4.coef_
        
        # AUC-ROC for logistic regression models
        r2_a[s,cluster] = r2_score(varB_norm,model1.predict(varA_norm))
        r2_b[s,cluster] = r2_score(varA_norm,model2.predict(varB_norm))
        auc_c[s, cluster] = roc_auc_score(varY, model3.predict_proba(varB_norm)[:, 1])
        auc_d[s, cluster] = roc_auc_score(varY, model4.predict_proba(varA_norm)[:, 1])
        
        # Indirect model
        model5 = LogisticRegression().fit(varAB_norm, varY)
        predicted_probs = model5.predict_proba(varAB_norm)[:, 1]
        
        e_all[s, cluster], f_all[s, cluster] = model5.coef_[0]
        
        # Partial AUC-ROC for e (controlling for varA)
        auc_e[s, cluster] = roc_auc_score(varY, predicted_probs, max_fpr=0.2)
        
        # Partial AUC-ROC for f (controlling for varB)
        auc_f[s, cluster] = roc_auc_score(varY, predicted_probs, max_fpr=0.2)
        
scores = np.array([r2_a,r2_b,auc_c,auc_d,auc_e, auc_f])
mean_scores_alpha = np.nanmean(scores,axis=1)

df = 24 - 1 -1 #subtracting one because s = 16 has no data
alpha = 0.05

#PLOT
names = ['Fig7A_1','Fig7A_2']
for cluster in range(2):
    p1 = ttest_1samp(a_all[:,cluster],0,nan_policy='omit')[1]
    p2 = ttest_1samp(b_all[:,cluster],0,nan_policy='omit')[1]
    p3 = ttest_1samp(c_all[:,cluster],0,nan_policy='omit')[1]
    p4 = ttest_1samp(d_all[:,cluster],0,nan_policy='omit')[1]
    p5 = ttest_1samp(e_all[:,cluster],0,nan_policy='omit')[1]
    p6 = ttest_1samp(f_all[:,cluster],0,nan_policy='omit')[1]
    
    mediation_model1 = np.nanmean(c_all[:,cluster] - e_all[:,cluster])
    se_model1 = np.nanstd(c_all[:,cluster] - e_all[:,cluster])/np.sqrt(len(c_all[:,cluster] - e_all[:,cluster]))
    zscore_model1 = mediation_model1 / se_model1
    p7 = stats.t.sf(zscore_model1, df)
    
    mediation_model2 = np.nanmean(d_all[:,cluster] - f_all[:,cluster])
    se_model2 = np.nanstd(d_all[:,cluster] - f_all[:,cluster])/np.sqrt(len(d_all[:,cluster] - f_all[:,cluster]))
    zscore_model2 = mediation_model2 / se_model2
    p8 = stats.t.sf(zscore_model2, df)
    
    fig, ax = plt.subplots(1, 1, figsize = (6, 2.5))
    data = np.array([np.delete(a_all[:,cluster],16),np.delete(b_all[:,cluster],16),np.delete(c_all[:,cluster],16),np.delete(d_all[:,cluster],16),np.delete(e_all[:,cluster],16),np.delete(f_all[:,cluster],16)])
    df = pd.DataFrame(data).transpose()
    df.columns = ["a","b","c","d","e","f"]
    df.to_csv(data_dir + names[cluster] + ".csv", index = False)

    ax.spines['left'].set_position(('outward', 10))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.spines['bottom'].set_position(('outward', 15))
    ax.xaxis.set_ticks_position('bottom')
    ax.axes.xaxis.set_ticks([])
    box1 = plt.boxplot(np.transpose(data), positions = [0,1,2,3,4,5], patch_artist = True, widths = 0.4,showfliers=False,boxprops=None,    showbox=None,     whis = 0, showcaps = False)
    box1['boxes'][0].set( facecolor = 'red', lw=0, zorder=0)
    box1['boxes'][1].set( facecolor = 'royalblue', lw=0, zorder=0)
    box1['boxes'][2].set( facecolor = 'mediumseagreen', lw=0, zorder=0)
    box1['boxes'][3].set( facecolor = 'gold', lw=0, zorder=0)
    box1['boxes'][4].set( facecolor = 'orchid', lw=0, zorder=0)
    box1['boxes'][5].set( facecolor = 'rebeccapurple', lw=0, zorder=0)
    box1['medians'][0].set( color = 'black', lw=2, zorder=20)
    box1['medians'][1].set( color = 'black', lw=2, zorder=20)
    box1['medians'][2].set( color = 'black', lw=2, zorder=20)
    box1['medians'][3].set( color = 'black', lw=2, zorder=20)
    box1['medians'][4].set( color = 'black', lw=2, zorder=20)
    box1['medians'][5].set( color = 'black', lw=2, zorder=20)
    plt.plot([-0.5,5.5], [0, 0], '--k')  
    plt.scatter(np.random.normal(loc = 0., scale = 0.08, size = 23), 
                data[0,:], s = 50, color = 'red', 
                edgecolor = 'black', zorder = 20)
    plt.scatter(np.random.normal(loc = 1., scale = 0.08, size = 23), 
                data[1,:], s = 50, color = 'royalblue', 
                edgecolor = 'black', zorder = 20)
    plt.scatter(np.random.normal(loc = 2., scale = 0.08, size = 23), 
                data[2,:], s = 50, color = 'mediumseagreen', 
                edgecolor = 'black', zorder = 20)
    plt.scatter(np.random.normal(loc = 3., scale = 0.08, size = 23), 
                data[3,:], s = 50, color = 'gold', 
                edgecolor = 'black', zorder = 20)
    plt.scatter(np.random.normal(loc = 4., scale = 0.08, size = 23), 
                data[4,:], s = 50, color = 'orchid', 
                edgecolor = 'black', zorder = 20)     
    plt.scatter(np.random.normal(loc = 5., scale = 0.08, size = 23), 
                data[5,:], s = 50, color = 'rebeccapurple', 
                edgecolor = 'black', zorder = 20)   
    plt.xticks(range(6), [r'$\mathbf{a}$', r'$\mathbf{b}$', r'$\mathbf{c}$', r'$\mathbf{d}$', r'$\mathbf{e}$', r'$\mathbf{f}$'], rotation=0, fontsize=18)
    plt.setp(ax.get_xticklabels()[0], color='red')
    plt.setp(ax.get_xticklabels()[1], color='royalblue')
    plt.setp(ax.get_xticklabels()[2], color='mediumseagreen')
    plt.setp(ax.get_xticklabels()[3], color='gold')
    plt.setp(ax.get_xticklabels()[4], color='orchid')
    plt.setp(ax.get_xticklabels()[5], color='rebeccapurple')
    plt.ylabel('beta coefficients',fontsize=18)
    maxY = np.nanmax(data[5,:])
    minY = np.nanmin(data[5,:])
    plt.ylim([minY-0.1, maxY+1]);
    
    if (p1 < 0.05) & (p1 >= 0.01):
        ax.annotate('*', xy=(0, maxY + 0.1), ha = 'center',fontsize=16)
    elif (p1 < 0.01) & (p1 >= 0.001):
        ax.annotate('**', xy=(0, maxY + 0.1), ha = 'center',fontsize=16)
    elif p1 < 0.001:
        ax.annotate('***', xy=(0, maxY + 0.1), ha = 'center',fontsize=16)
    else:
        ax.annotate('n.s.', xy=(0, maxY + 0.1), ha = 'center',fontsize=16)
        
    if (p2 < 0.05) & (p2 >= 0.01):
        ax.annotate('*', xy=(1, maxY + 0.1), ha = 'center',fontsize=16)
    elif (p2 < 0.01) & (p2 >= 0.001):
        ax.annotate('**', xy=(1, maxY + 0.1), ha = 'center',fontsize=16)
    elif p2 < 0.001:
        ax.annotate('***', xy=(1, maxY + 0.1), ha = 'center',fontsize=16)
    else:
        ax.annotate('n.s.', xy=(1, maxY + 0.1), ha = 'center',fontsize=16)
        
    if (p3 < 0.05) & (p3 >= 0.01):
        ax.annotate('*', xy=(2, maxY + 0.1), ha = 'center',fontsize=16)
    elif (p3 < 0.01) & (p3 >= 0.001):
        ax.annotate('**', xy=(2, maxY + 0.1), ha = 'center',fontsize=16)
    elif p3 < 0.001:
        ax.annotate('***', xy=(2, maxY + 0.1), ha = 'center',fontsize=16)
    else:
        ax.annotate('n.s.', xy=(2, maxY + 0.1), ha = 'center',fontsize=16)
        
    if (p4 < 0.05) & (p4 >= 0.01):
        ax.annotate('*', xy=(3, maxY + 0.1), ha = 'center',fontsize=16)
    elif (p4 < 0.01) & (p4 >= 0.001):
        ax.annotate('**', xy=(3, maxY + 0.1), ha = 'center',fontsize=16)
    elif p4 < 0.001:
        ax.annotate('***', xy=(3, maxY + 0.1), ha = 'center',fontsize=16)
    else:
        ax.annotate('n.s.', xy=(3, maxY + 0.1), ha = 'center',fontsize=16)
        
    if (p5 < 0.05) & (p5 >= 0.01):
        ax.annotate('*', xy=(4, maxY + 0.1), ha = 'center',fontsize=16)
    elif (p5 < 0.01) & (p5 >= 0.001):
        ax.annotate('**', xy=(4, maxY + 0.1), ha = 'center',fontsize=16)
    elif p5 < 0.001:
        ax.annotate('***', xy=(4, maxY + 0.1), ha = 'center',fontsize=16)
    else:
        ax.annotate('n.s.', xy=(4, maxY + 0.1), ha = 'center',fontsize=16)
        
    if (p6 < 0.05) & (p6 >= 0.01):
        ax.annotate('*', xy=(5, maxY + 0.1), ha = 'center',fontsize=16)
    elif (p6 < 0.01) & (p6 >= 0.001):
        ax.annotate('**', xy=(5, maxY + 0.1), ha = 'center',fontsize=16)
    elif p6 < 0.001:
        ax.annotate('***', xy=(5, maxY + 0.1), ha = 'center',fontsize=16)
    else:
        ax.annotate('n.s.', xy=(5, maxY + 0.1), ha = 'center',fontsize=16)
    
    if (p7 < 0.001):
        ax.annotate('p < 0.001', xy=(3, maxY + 0.53), zorder=10, ha = 'center',fontsize=18)
    else:
        ax.annotate('p = ' + str(round(p7,3)), xy=(3, maxY + 0.53), zorder=10, ha = 'center',fontsize=18)
    props = {'connectionstyle':'bar','arrowstyle':'-',\
                     'shrinkA':25,'shrinkB':25}
    ax.annotate('', xy=(2, maxY+0.02), xytext=(4, maxY+0.02), arrowprops=props)
    
    if (p8 < 0.001):
        ax.annotate('p < 0.001', xy=(4, maxY + 0.95), zorder=10, ha = 'center',fontsize=18)
    else:
        ax.annotate('p = ' + str(round(p8,2)), xy=(4, maxY + 0.95), zorder=10, ha = 'center',fontsize=18)
    props = {'connectionstyle':'bar','arrowstyle':'-',\
                     'shrinkA':25,'shrinkB':25}
    ax.annotate('', xy=(3, maxY+0.43), xytext=(5, maxY+0.43), arrowprops=props)
    
    plt.savefig(figures_dir + names[cluster] + ".png",dpi=800, bbox_inches='tight',transparent=True)
    plt.clf()


## TESTING MEDIATION OF BETA ON RECOGNITION VIA PUPIL

#GLM1: pupil (M) = i + a * beta (X) + err
#GLM2: beta = i + b * pupil + err
#GLM3: recognition (Y) = i + c * pupil (M) + err
#GLM4: recognition (Y) = i + d * beta (X) + err
#GLM5: recognition (Y) = i + f * beta (X) + e * pupil (M) + err

a_all = np.zeros((len(HLTP.subjects),2))
b_all = np.zeros((len(HLTP.subjects),2))
c_all = np.zeros((len(HLTP.subjects),2))
d_all = np.zeros((len(HLTP.subjects),2))
e_all = np.zeros((len(HLTP.subjects),2))
f_all = np.zeros((len(HLTP.subjects),2))

r2_a = np.zeros((len(HLTP.subjects),2))
r2_b = np.zeros((len(HLTP.subjects),2))
auc_c = np.zeros((len(HLTP.subjects),2))
auc_d = np.zeros((len(HLTP.subjects),2))
auc_e = np.zeros((len(HLTP.subjects),2))
auc_f = np.zeros((len(HLTP.subjects),2))

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score
from scipy.stats import zscore
import scipy.stats as stats

for s, sub in enumerate(HLTP.subjects):
    if s == 16:
        a_all[s,:] = np.nan
        b_all[s,:] = np.nan
        c_all[s,:] = np.nan
        d_all[s,:] = np.nan
        e_all[s,:] = np.nan
        f_all[s,:] = np.nan
        r2_a[s] = np.nan
        r2_b[s] = np.nan
        auc_c[s] = np.nan
        auc_d[s] = np.nan
        auc_e[s] = np.nan
        auc_f[s] = np.nan
        continue
    for cluster in range(len(all_clusters[2:4])):
        df = alldata_allsubs[(alldata_allsubs.subjID == sub) & (alldata_allsubs.clusterID == (cluster+2))]
        df_clean = df.dropna()
        
        varA = df_clean.beta.values.reshape(-1, 1)
        varB = df_clean.PupilSize.values.reshape(-1,1)
        varY = df_clean.BehavRecognition.values.reshape(-1,1)
        varAB = df_clean[['PupilSize','beta']].values
        
        #Normalize z-score
        varA_norm = zscore(varA)
        varB_norm = zscore(varB)
        varAB_norm = np.hstack((varA_norm,varB_norm))
        
        # Direct models
        model1 = LinearRegression().fit(varA_norm, varB_norm)
        a_all[s, cluster] = model1.coef_
        
        model2 = LinearRegression().fit(varB_norm, varA_norm)
        b_all[s, cluster] = model2.coef_
        
        model3 = LogisticRegression().fit(varB_norm, varY)
        c_all[s, cluster] = model3.coef_
        
        model4 = LogisticRegression().fit(varA_norm, varY)
        d_all[s, cluster] = model4.coef_
        
        # AUC-ROC for logistic regression models
        r2_a[s,cluster] = r2_score(varB_norm,model1.predict(varA_norm))
        r2_b[s,cluster] = r2_score(varA_norm,model2.predict(varB_norm))
        auc_c[s, cluster] = roc_auc_score(varY, model3.predict_proba(varB_norm)[:, 1])
        auc_d[s, cluster] = roc_auc_score(varY, model4.predict_proba(varA_norm)[:, 1])
        
        # Indirect model
        model5 = LogisticRegression().fit(varAB_norm, varY)
        predicted_probs = model5.predict_proba(varAB_norm)[:, 1]
        
        e_all[s, cluster], f_all[s, cluster] = model5.coef_[0]
        
        # Partial AUC-ROC for e (controlling for varA)
        auc_e[s, cluster] = roc_auc_score(varY, predicted_probs, max_fpr=0.2)
        
        # Partial AUC-ROC for f (controlling for varB)
        auc_f[s, cluster] = roc_auc_score(varY, predicted_probs, max_fpr=0.2)
        
scores = np.array([r2_a,r2_b,auc_c,auc_d,auc_e, auc_f])
mean_scores_beta = np.nanmean(scores,axis=1)

df = 24 - 1 -1 #subtracting one because s = 16 has no data
alpha = 0.05

#PLOT
names = ['Fig7B_1','Fig7B_2']
for cluster in range(2):
    p1 = ttest_1samp(a_all[:,cluster],0,nan_policy='omit')[1]
    p2 = ttest_1samp(b_all[:,cluster],0,nan_policy='omit')[1]
    p3 = ttest_1samp(c_all[:,cluster],0,nan_policy='omit')[1]
    p4 = ttest_1samp(d_all[:,cluster],0,nan_policy='omit')[1]
    p5 = ttest_1samp(e_all[:,cluster],0,nan_policy='omit')[1]
    p6 = ttest_1samp(f_all[:,cluster],0,nan_policy='omit')[1]
    
    mediation_model1 = np.nanmean(c_all[:,cluster] - e_all[:,cluster])
    se_model1 = np.nanstd(c_all[:,cluster] - e_all[:,cluster])/np.sqrt(len(c_all[:,cluster] - e_all[:,cluster]))
    zscore_model1 = mediation_model1 / se_model1
    p7 = stats.t.sf(zscore_model1, df)

    mediation_model2 = np.nanmean(d_all[:,cluster] - f_all[:,cluster])
    se_model2 = np.nanstd(d_all[:,cluster] - f_all[:,cluster])/np.sqrt(len(d_all[:,cluster] - f_all[:,cluster]))
    zscore_model2 = mediation_model2 / se_model2
    p8 = stats.t.sf(zscore_model2, df)
    
    fig, ax = plt.subplots(1, 1, figsize = (6, 2.5))
    data = np.array([np.delete(a_all[:,cluster],16),np.delete(b_all[:,cluster],16),np.delete(c_all[:,cluster],16),np.delete(d_all[:,cluster],16),np.delete(e_all[:,cluster],16),np.delete(f_all[:,cluster],16)])
    df = pd.DataFrame(data).transpose()
    df.columns = ["a","b","c","d","e","f"]
    df.to_csv(data_dir + names[cluster] + ".csv", index = False)

    ax.spines['left'].set_position(('outward', 10))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.spines['bottom'].set_position(('outward', 15))
    ax.xaxis.set_ticks_position('bottom')
    ax.axes.xaxis.set_ticks([])
    box1 = plt.boxplot(np.transpose(data), positions = [0,1,2,3,4,5], patch_artist = True, widths = 0.4,showfliers=False,boxprops=None,    showbox=None,     whis = 0, showcaps = False)
    box1['boxes'][0].set( facecolor = 'red', lw=0, zorder=0)
    box1['boxes'][1].set( facecolor = 'royalblue', lw=0, zorder=0)
    box1['boxes'][2].set( facecolor = 'mediumseagreen', lw=0, zorder=0)
    box1['boxes'][3].set( facecolor = 'gold', lw=0, zorder=0)
    box1['boxes'][4].set( facecolor = 'orchid', lw=0, zorder=0)
    box1['boxes'][5].set( facecolor = 'rebeccapurple', lw=0, zorder=0)
    box1['medians'][0].set( color = 'black', lw=2, zorder=20)
    box1['medians'][1].set( color = 'black', lw=2, zorder=20)
    box1['medians'][2].set( color = 'black', lw=2, zorder=20)
    box1['medians'][3].set( color = 'black', lw=2, zorder=20)
    box1['medians'][4].set( color = 'black', lw=2, zorder=20)
    box1['medians'][5].set( color = 'black', lw=2, zorder=20)
    plt.plot([-0.5,5.5], [0, 0], '--k')  
    plt.scatter(np.random.normal(loc = 0., scale = 0.08, size = 23), 
                data[0,:], s = 50, color = 'red', 
                edgecolor = 'black', zorder = 20)
    plt.scatter(np.random.normal(loc = 1., scale = 0.08, size = 23), 
                data[1,:], s = 50, color = 'royalblue', 
                edgecolor = 'black', zorder = 20)
    plt.scatter(np.random.normal(loc = 2., scale = 0.08, size = 23), 
                data[2,:], s = 50, color = 'mediumseagreen', 
                edgecolor = 'black', zorder = 20)
    plt.scatter(np.random.normal(loc = 3., scale = 0.08, size = 23), 
                data[3,:], s = 50, color = 'gold', 
                edgecolor = 'black', zorder = 20)
    plt.scatter(np.random.normal(loc = 4., scale = 0.08, size = 23), 
                data[4,:], s = 50, color = 'orchid', 
                edgecolor = 'black', zorder = 20)     
    plt.scatter(np.random.normal(loc = 5., scale = 0.08, size = 23), 
                data[5,:], s = 50, color = 'rebeccapurple', 
                edgecolor = 'black', zorder = 20)   
    plt.xticks(range(6), [r'$\mathbf{a}$', r'$\mathbf{b}$', r'$\mathbf{c}$', r'$\mathbf{d}$', r'$\mathbf{e}$', r'$\mathbf{f}$'], rotation=0, fontsize=18)
    plt.setp(ax.get_xticklabels()[0], color='red')
    plt.setp(ax.get_xticklabels()[1], color='royalblue')
    plt.setp(ax.get_xticklabels()[2], color='mediumseagreen')
    plt.setp(ax.get_xticklabels()[3], color='gold')
    plt.setp(ax.get_xticklabels()[4], color='orchid')
    plt.setp(ax.get_xticklabels()[5], color='rebeccapurple')
    plt.ylabel('beta coefficients',fontsize=18)
    maxY = np.nanmax(data[5,:])
    minY = np.nanmin(data[5,:])
    plt.ylim([minY-0.1, maxY+1]);
    
    if (p1 < 0.05) & (p1 >= 0.01):
        ax.annotate('*', xy=(0, maxY + 0.1), ha = 'center',fontsize=16)
    elif (p1 < 0.01) & (p1 >= 0.001):
        ax.annotate('**', xy=(0, maxY + 0.1), ha = 'center',fontsize=16)
    elif p1 < 0.001:
        ax.annotate('***', xy=(0, maxY + 0.1), ha = 'center',fontsize=16)
    else:
        ax.annotate('n.s.', xy=(0, maxY + 0.1), ha = 'center',fontsize=16)
        
    if (p2 < 0.05) & (p2 >= 0.01):
        ax.annotate('*', xy=(1, maxY + 0.1), ha = 'center',fontsize=16)
    elif (p2 < 0.01) & (p2 >= 0.001):
        ax.annotate('**', xy=(1, maxY + 0.1), ha = 'center',fontsize=16)
    elif p2 < 0.001:
        ax.annotate('***', xy=(1, maxY + 0.1), ha = 'center',fontsize=16)
    else:
        ax.annotate('n.s.', xy=(1, maxY + 0.1), ha = 'center',fontsize=16)
        
    if (p3 < 0.05) & (p3 >= 0.01):
        ax.annotate('*', xy=(2, maxY + 0.1), ha = 'center',fontsize=16)
    elif (p3 < 0.01) & (p3 >= 0.001):
        ax.annotate('**', xy=(2, maxY + 0.1), ha = 'center',fontsize=16)
    elif p3 < 0.001:
        ax.annotate('***', xy=(2, maxY + 0.1), ha = 'center',fontsize=16)
    else:
        ax.annotate('n.s.', xy=(2, maxY + 0.1), ha = 'center',fontsize=16)
        
    if (p4 < 0.05) & (p4 >= 0.01):
        ax.annotate('*', xy=(3, maxY + 0.1), ha = 'center',fontsize=16)
    elif (p4 < 0.01) & (p4 >= 0.001):
        ax.annotate('**', xy=(3, maxY + 0.1), ha = 'center',fontsize=16)
    elif p4 < 0.001:
        ax.annotate('***', xy=(3, maxY + 0.1), ha = 'center',fontsize=16)
    else:
        ax.annotate('n.s.', xy=(3, maxY + 0.1), ha = 'center',fontsize=16)
        
    if (p5 < 0.05) & (p5 >= 0.01):
        ax.annotate('*', xy=(4, maxY + 0.1), ha = 'center',fontsize=16)
    elif (p5 < 0.01) & (p5 >= 0.001):
        ax.annotate('**', xy=(4, maxY + 0.1), ha = 'center',fontsize=16)
    elif p5 < 0.001:
        ax.annotate('***', xy=(4, maxY + 0.1), ha = 'center',fontsize=16)
    else:
        ax.annotate('n.s.', xy=(4, maxY + 0.1), ha = 'center',fontsize=16)
        
    if (p6 < 0.05) & (p6 >= 0.01):
        ax.annotate('*', xy=(5, maxY + 0.1), ha = 'center',fontsize=16)
    elif (p6 < 0.01) & (p6 >= 0.001):
        ax.annotate('**', xy=(5, maxY + 0.1), ha = 'center',fontsize=16)
    elif p6 < 0.001:
        ax.annotate('***', xy=(5, maxY + 0.1), ha = 'center',fontsize=16)
    else:
        ax.annotate('n.s.', xy=(5, maxY + 0.1), ha = 'center',fontsize=16)
    
    if (p7 < 0.001):
        ax.annotate('p < 0.001', xy=(3, maxY + 0.53), zorder=10, ha = 'center',fontsize=18)
    else:
        ax.annotate('p = ' + str(round(p7,3)), xy=(3, maxY + 0.53), zorder=10, ha = 'center',fontsize=18)
    props = {'connectionstyle':'bar','arrowstyle':'-',\
                     'shrinkA':25,'shrinkB':25}
    ax.annotate('', xy=(2, maxY+0.02), xytext=(4, maxY+0.02), arrowprops=props)
    
    if (p8 < 0.001):
        ax.annotate('p < 0.001', xy=(4, maxY + 0.95), zorder=10, ha = 'center',fontsize=18)
    else:
        ax.annotate('p = ' + str(round(p8,2)), xy=(4, maxY + 0.95), zorder=10, ha = 'center',fontsize=18)
    props = {'connectionstyle':'bar','arrowstyle':'-',\
                     'shrinkA':25,'shrinkB':25}
    ax.annotate('', xy=(3, maxY+0.43), xytext=(5, maxY+0.43), arrowprops=props)
    
    plt.savefig(figures_dir + names[cluster] + ".png",dpi=800, bbox_inches='tight',transparent=True)
    plt.clf()
    
## TESTING MEDIATION OF SCP ON RECOGNITION VIA PUPIL
## For this one don't use clusters -- use estimates across entire pre-stimulus time window.    
#GLM1: pupil (M) = i + a * SCP (X) + e
#GLM2: SCP = b * pupil + e
#GLM2: recognition (Y) = i + c * pupil (M) + e
#GLM3: recognition (Y) = i + d * SCP (X) + e
#GLM4: recognition (Y) = i + f * SCP (X) + e * pupil (M) + e
a_all = np.zeros((len(HLTP.subjects)))
b_all = np.zeros((len(HLTP.subjects)))
c_all = np.zeros((len(HLTP.subjects)))
d_all = np.zeros((len(HLTP.subjects)))
e_all = np.zeros((len(HLTP.subjects)))
f_all = np.zeros((len(HLTP.subjects)))

r2_a = np.zeros((len(HLTP.subjects)))
r2_b = np.zeros((len(HLTP.subjects)))
auc_c = np.zeros((len(HLTP.subjects)))
auc_d = np.zeros((len(HLTP.subjects)))
auc_e = np.zeros((len(HLTP.subjects)))
auc_f = np.zeros((len(HLTP.subjects)))

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score
from scipy.stats import zscore
import scipy.stats as stats

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
        d_all[s] = np.nan
        e_all[s] = np.nan
        f_all[s] = np.nan
        r2_a[s] = np.nan
        r2_b[s] = np.nan
        auc_c[s] = np.nan
        auc_d[s] = np.nan
        auc_e[s] = np.nan
        auc_f[s] = np.nan
        continue
    df = allSCP_pupildata[(allSCP_pupildata.subjID == sub)]
    df_clean = df.dropna()
    
    varA = df_clean.pred_prob.values.reshape(-1, 1)
    varB = df_clean.pupil_size.values.reshape(-1,1)
    varY = df_clean.BehavRecognition.values.reshape(-1,1)
    varY = np.array(varY, dtype=bool)
    varAB = df_clean[['pupil_size','pred_prob']].values
    
    #Normalize z-score
    varA_norm = zscore(varA)
    varB_norm = zscore(varB)
    varAB_norm = np.hstack((varA_norm,varB_norm))
    
    # Direct models
    model1 = LinearRegression().fit(varA_norm, varB_norm)
    a_all[s] = model1.coef_
    
    model2 = LinearRegression().fit(varB_norm, varA_norm)
    b_all[s] = model2.coef_
    
    model3 = LogisticRegression().fit(varB_norm, varY)
    c_all[s] = model3.coef_
    
    model4 = LogisticRegression().fit(varA_norm, varY)
    d_all[s] = model4.coef_
    
    # AUC-ROC for logistic regression models
    r2_a[s] = r2_score(varB_norm,model1.predict(varA_norm))
    r2_b[s] = r2_score(varA_norm,model2.predict(varB_norm))
    auc_c[s] = roc_auc_score(varY, model3.predict_proba(varB_norm)[:, 1])
    auc_d[s] = roc_auc_score(varY, model4.predict_proba(varA_norm)[:, 1])
    
    # Indirect model
    model5 = LogisticRegression().fit(varAB_norm, varY)
    predicted_probs = model5.predict_proba(varAB_norm)[:, 1]
    
    e_all[s], f_all[s] = model5.coef_[0]
    
    # Partial AUC-ROC for e (controlling for varA)
    auc_e[s] = roc_auc_score(varY, predicted_probs, max_fpr=0.2)
    
    # Partial AUC-ROC for f (controlling for varB)
    auc_f[s] = roc_auc_score(varY, predicted_probs, max_fpr=0.2)

scores = np.array([r2_a,r2_b,auc_c,auc_d,auc_e,auc_f])
mean_scores_scp = np.nanmean(scores,axis=1)

## Plotting
alpha = 0.05
df = 22

p1 = ttest_1samp(a_all[:],0,nan_policy='omit')[1]
p2 = ttest_1samp(b_all[:],0,nan_policy='omit')[1]
p3 = ttest_1samp(c_all[:],0,nan_policy='omit')[1]
p4 = ttest_1samp(d_all[:],0,nan_policy='omit')[1]
p5 = ttest_1samp(e_all[:],0,nan_policy='omit')[1]
p6 = ttest_1samp(f_all[:],0,nan_policy='omit')[1]

mediation_model1 = np.nanmean(c_all - e_all)
se_model1 = np.nanstd(c_all - e_all)/np.sqrt(len(c_all - e_all))
zscore_model1 = mediation_model1 / se_model1
p7 = stats.t.sf(zscore_model1, df)

mediation_model2 = np.nanmean(d_all - f_all)
se_model2 = np.nanstd(d_all - f_all)/np.sqrt(len(d_all - f_all))
zscore_model2 = mediation_model2 / se_model2
p8 = stats.t.sf(zscore_model2, df)

fig, ax = plt.subplots(1, 1, figsize = (6, 2.5))
data = np.array([np.delete(a_all,16),np.delete(b_all,16),np.delete(c_all,16),np.delete(d_all,16),np.delete(e_all,16),np.delete(f_all,16)])
df = pd.DataFrame(data).transpose()
df.columns = ["a","b","c","d","e","f"]
df.to_csv(data_dir + "Fig7C.csv", index = False)

ax.spines['left'].set_position(('outward', 10))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.spines['bottom'].set_position(('outward', 15))
ax.xaxis.set_ticks_position('bottom')
ax.axes.xaxis.set_ticks([])
box1 = plt.boxplot(np.transpose(data), positions = [0,1,2,3,4,5], patch_artist = True, widths = 0.4,showfliers=False,boxprops=None,    showbox=None,     whis = 0, showcaps = False)
box1['boxes'][0].set( facecolor = 'red', lw=0, zorder=0)
box1['boxes'][1].set( facecolor = 'royalblue', lw=0, zorder=0)
box1['boxes'][2].set( facecolor = 'mediumseagreen', lw=0, zorder=0)
box1['boxes'][3].set( facecolor = 'gold', lw=0, zorder=0)
box1['boxes'][4].set( facecolor = 'orchid', lw=0, zorder=0)
box1['boxes'][5].set( facecolor = 'rebeccapurple', lw=0, zorder=0)
box1['medians'][0].set( color = 'black', lw=2, zorder=20)
box1['medians'][1].set( color = 'black', lw=2, zorder=20)
box1['medians'][2].set( color = 'black', lw=2, zorder=20)
box1['medians'][3].set( color = 'black', lw=2, zorder=20)
box1['medians'][4].set( color = 'black', lw=2, zorder=20)
box1['medians'][5].set( color = 'black', lw=2, zorder=20)
plt.plot([-0.5,5.5], [0, 0], '--k')  
plt.scatter(np.random.normal(loc = 0., scale = 0.08, size = 23), 
            data[0,:], s = 50, color = 'red', 
            edgecolor = 'black', zorder = 20)
plt.scatter(np.random.normal(loc = 1., scale = 0.08, size = 23), 
            data[1,:], s = 50, color = 'royalblue', 
            edgecolor = 'black', zorder = 20)
plt.scatter(np.random.normal(loc = 2., scale = 0.08, size = 23), 
            data[2,:], s = 50, color = 'mediumseagreen', 
            edgecolor = 'black', zorder = 20)
plt.scatter(np.random.normal(loc = 3., scale = 0.08, size = 23), 
            data[3,:], s = 50, color = 'gold', 
            edgecolor = 'black', zorder = 20)
plt.scatter(np.random.normal(loc = 4., scale = 0.08, size = 23), 
            data[4,:], s = 50, color = 'orchid', 
            edgecolor = 'black', zorder = 20)     
plt.scatter(np.random.normal(loc = 5., scale = 0.08, size = 23), 
            data[5,:], s = 50, color = 'rebeccapurple', 
            edgecolor = 'black', zorder = 20)   
plt.xticks(range(6), [r'$\mathbf{a}$', r'$\mathbf{b}$', r'$\mathbf{c}$', r'$\mathbf{d}$', r'$\mathbf{e}$', r'$\mathbf{f}$'], rotation=0, fontsize=18)
plt.setp(ax.get_xticklabels()[0], color='red')
plt.setp(ax.get_xticklabels()[1], color='royalblue')
plt.setp(ax.get_xticklabels()[2], color='mediumseagreen')
plt.setp(ax.get_xticklabels()[3], color='gold')
plt.setp(ax.get_xticklabels()[4], color='orchid')
plt.setp(ax.get_xticklabels()[5], color='rebeccapurple')
plt.ylabel('beta coefficients',fontsize=18)
maxY = np.nanmax(data[3,:])
minY = np.nanmin(data[0,:])
plt.ylim([minY-0.1, maxY+1.5]);

if (p1 < 0.05) & (p1 >= 0.01):
    ax.annotate('*', xy=(0, maxY + 0.1), ha = 'center',fontsize=16)
elif (p1 < 0.01) & (p1 >= 0.001):
    ax.annotate('**', xy=(0, maxY + 0.1), ha = 'center',fontsize=16)
elif p1 < 0.001:
    ax.annotate('***', xy=(0, maxY + 0.1), ha = 'center',fontsize=16)
else:
    ax.annotate('n.s.', xy=(0, maxY + 0.1), ha = 'center',fontsize=16)
    
if (p2 < 0.05) & (p2 >= 0.01):
    ax.annotate('*', xy=(1, maxY + 0.1), ha = 'center',fontsize=16)
elif (p2 < 0.01) & (p2 >= 0.001):
    ax.annotate('**', xy=(1, maxY + 0.1), ha = 'center',fontsize=16)
elif p2 < 0.001:
    ax.annotate('***', xy=(1, maxY + 0.1), ha = 'center',fontsize=16)
else:
    ax.annotate('n.s.', xy=(1, maxY + 0.1), ha = 'center',fontsize=16)
    
if (p3 < 0.05) & (p3 >= 0.01):
    ax.annotate('*', xy=(2, maxY + 0.1), ha = 'center',fontsize=16)
elif (p3 < 0.01) & (p3 >= 0.001):
    ax.annotate('**', xy=(2, maxY + 0.1), ha = 'center',fontsize=16)
elif p3 < 0.001:
    ax.annotate('***', xy=(2, maxY + 0.1), ha = 'center',fontsize=16)
else:
    ax.annotate('n.s.', xy=(2, maxY + 0.1), ha = 'center',fontsize=16)
    
if (p4 < 0.05) & (p4 >= 0.01):
    ax.annotate('*', xy=(3, maxY + 0.1), ha = 'center',fontsize=16)
elif (p4 < 0.01) & (p4 >= 0.001):
    ax.annotate('**', xy=(3, maxY + 0.1), ha = 'center',fontsize=16)
elif p4 < 0.001:
    ax.annotate('***', xy=(3, maxY + 0.1), ha = 'center',fontsize=16)
else:
    ax.annotate('n.s.', xy=(3, maxY + 0.1), ha = 'center',fontsize=16)
    
if (p5 < 0.05) & (p5 >= 0.01):
    ax.annotate('*', xy=(4, maxY + 0.1), ha = 'center',fontsize=16)
elif (p5 < 0.01) & (p5 >= 0.001):
    ax.annotate('**', xy=(4, maxY + 0.1), ha = 'center',fontsize=16)
elif p5 < 0.001:
    ax.annotate('***', xy=(4, maxY + 0.1), ha = 'center',fontsize=16)
else:
    ax.annotate('n.s.', xy=(4, maxY + 0.1), ha = 'center',fontsize=16)
    
if (p6 < 0.05) & (p6 >= 0.01):
    ax.annotate('*', xy=(5, maxY + 0.1), ha = 'center',fontsize=16)
elif (p6 < 0.01) & (p6 >= 0.001):
    ax.annotate('**', xy=(5, maxY + 0.1), ha = 'center',fontsize=16)
elif p6 < 0.001:
    ax.annotate('***', xy=(5, maxY + 0.1), ha = 'center',fontsize=16)
else:
    ax.annotate('n.s.', xy=(5, maxY + 0.1), ha = 'center',fontsize=16)

if (p7 < 0.05) & (p7 >= 0.01):
    ax.annotate('*', xy=(6, maxY + 0.1), ha = 'center',fontsize=16)
elif (p7 < 0.01) & (p7 >= 0.001):
    ax.annotate('**', xy=(6, maxY + 0.1), ha = 'center',fontsize=16)
elif p7 < 0.001:
    ax.annotate('***', xy=(6, maxY + 0.1), ha = 'center',fontsize=16)
else:
    ax.annotate('n.s.', xy=(6, maxY + 0.1), ha = 'center',fontsize=16)
    
if (p7 < 0.001):
    ax.annotate('p < 0.001', xy=(3, maxY + 0.67), zorder=10, ha = 'center',fontsize=18)
else:
    ax.annotate('p = ' + str(round(p7,3)), xy=(3, maxY + 0.7), zorder=10, ha = 'center',fontsize=18)
props = {'connectionstyle':'bar','arrowstyle':'-',\
                 'shrinkA':25,'shrinkB':25}
ax.annotate('', xy=(2, maxY-0.12), xytext=(4, maxY-0.12), arrowprops=props)

if (p8 < 0.001):
    ax.annotate('p < 0.001', xy=(4, maxY + 1.33), zorder=10, ha = 'center',fontsize=18)
else:
    ax.annotate('p = ' + str(round(p8,2)), xy=(4, maxY + 1.33), zorder=10, ha = 'center',fontsize=18)
props = {'connectionstyle':'bar','arrowstyle':'-',\
                 'shrinkA':25,'shrinkB':25}
ax.annotate('', xy=(3, maxY+0.52), xytext=(5, maxY+0.52), arrowprops=props)

plt.savefig(figures_dir + "Fig7C.png",dpi=800, bbox_inches='tight',transparent=True)
plt.clf()
    
## Make table with all R2 scores
r2_scores_all = np.vstack([np.transpose(mean_scores_alpha),np.transpose(mean_scores_beta),np.transpose(mean_scores_scp)])
df = pd.DataFrame(r2_scores_all)
# Save the DataFrame to a CSV file
df.to_csv(figures_dir + 'Fig7D.csv', index=False, header=False)
