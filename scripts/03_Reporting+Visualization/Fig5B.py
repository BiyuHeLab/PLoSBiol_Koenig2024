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
#import statsmodels.api as sm
box_style=dict(boxstyle='round', facecolor='wheat', alpha=0.5)

os.chdir('/isilon/LFMI/VMdrive/Lua/MEG_alpha_1-f_SCP_PLoSBiol_2024/scripts')
data_dir = '/isilon/LFMI/VMdrive/Lua/MEG_alpha_1-f_SCP_PLoSBiol_2024/data/HL/'
figures_dir = '/isilon/LFMI/VMdrive/Lua/MEG_alpha_1-f_SCP_PLoSBiol_2024/figures/'
supp_dir = '/isilon/LFMI/VMdrive/Lua/MEG_alpha_1-f_SCP_PLoSBiol_2024/scripts/supporting_files_toolboxes/HL/'

import HLTP

matalpha = scipy.io.loadmat(data_dir + 'alpha_cluster_sensors.mat')
matbeta = scipy.io.loadmat(data_dir + 'beta_cluster_sensors.mat')
alpha_clusters = matalpha['alpha_cluster_sensors'][0,:] - 1 #Subtracting 1 to account for Matlab indexing starting from 1 instead of 0
beta_clusters = matbeta['beta_cluster_sensors'][0,:] - 1 #Subtracting 1 to account for Matlab indexing starting from 1 instead of 0

all_clusters = np.concatenate((alpha_clusters,beta_clusters))
cluster_times_alpha = np.array([1,3])
cluster_times_beta = np.array([2,3])
cluster_times = np.concatenate([cluster_times_alpha,cluster_times_beta]) #time intervals at which each cluster was significant

alldata_allsubs = pd.read_pickle(data_dir + "alldata_allsubjects_cluster_withpupil.pkl")
allSCP_pupildata = pd.read_pickle(data_dir + "allpupil_SCP_data.pkl")
from scipy.special import logit
allSCP_pupildata['pred_prob'] = logit(allSCP_pupildata['pred_prob'])
allSCP_pupildata.replace([np.inf, -np.inf], np.nan, inplace=True)

cluster_corr_alphabeta_p = np.zeros((4,len(HLTP.subjects))) #alpha1 with beta 1, alpha 1 with beta 2, alpha 2 with beta 1, alpha 2 with beta 2
cluster_corr_alphabeta_rho = np.zeros((4,len(HLTP.subjects)))

for s, sub in enumerate(HLTP.subjects):
    df_a1 = alldata_allsubs[(alldata_allsubs.subjID == sub) & (alldata_allsubs.clusterID == 0)]
    df_a2 = alldata_allsubs[(alldata_allsubs.subjID == sub) & (alldata_allsubs.clusterID == 1)]
    df_b1 = alldata_allsubs[(alldata_allsubs.subjID == sub) & (alldata_allsubs.clusterID == 2)]
    df_b2 = alldata_allsubs[(alldata_allsubs.subjID == sub) & (alldata_allsubs.clusterID == 3)]
    
    cluster_corr_alphabeta_rho[0,s], cluster_corr_alphabeta_p[0,s] = spearmanr(df_a1[['alpha']],df_b1[['beta']],nan_policy='omit')
    cluster_corr_alphabeta_rho[1,s], cluster_corr_alphabeta_p[1,s] = spearmanr(df_a1[['alpha']],df_b2[['beta']],nan_policy='omit')
    cluster_corr_alphabeta_rho[2,s], cluster_corr_alphabeta_p[2,s] = spearmanr(df_a2[['alpha']],df_b1[['beta']],nan_policy='omit')
    cluster_corr_alphabeta_rho[3,s], cluster_corr_alphabeta_p[3,s] = spearmanr(df_a2[['alpha']],df_b2[['beta']],nan_policy='omit')
    
##Assessing significance of the correlations
corr_alphabeta_z = np.arctanh(cluster_corr_alphabeta_rho)

pvalues_alphabeta = np.zeros((4))
Wvalues_alphabeta = np.zeros((4))

for corrnum in range(4):
    Wvalues_alphabeta[corrnum],pvalues_alphabeta[corrnum] = wilcoxon(corr_alphabeta_z[corrnum,:])

# Barplots
box_style=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
colors = {'AlphaC1_high':np.array([164, 210, 255]) / 255., 
          'AlphaC2_high':np.array([4, 51, 255]) / 255., 
          'AlphaC1_low':np.array([255, 221, 140]) / 255.,
          'BetaC1_high':np.array([174, 236, 131]) / 255.,
          'BetaC2_high':np.array([53, 120, 33]) / 255., 
          'BetaC1_low':np.array([244, 170, 59]) / 255.,
          'BetaC2_low':np.array([255, 102, 102]) / 255.,
          'SCP_pupil':np.array([204, 204, 255]) / 255.}        


#Alpha vs. beta clusters
plt.clf()
data = [corr_alphabeta_z[0,:],corr_alphabeta_z[1,:],corr_alphabeta_z[2,:],corr_alphabeta_z[3,:]]
fig, ax = plt.subplots(1, 1, figsize = (6, 2.5))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('outward', 10))
ax.yaxis.set_ticks_position('left')
ax.spines['bottom'].set_position(('outward', 15))
ax.xaxis.set_ticks_position('bottom')
ax.axes.xaxis.set_ticks([])
box1 = plt.boxplot(data, positions = [0,1,2,3], patch_artist = True, 
               widths = 0.8,showfliers=False,
     boxprops=None,    showbox=None,     whis = 0, showcaps = False)
box1['boxes'][0].set( facecolor = 'firebrick', lw=0, zorder=0)
box1['boxes'][1].set( facecolor = 'crimson', lw=0, zorder=0)
box1['boxes'][2].set( facecolor = 'indianred', lw=0, zorder=0)
box1['boxes'][3].set( facecolor = 'lightcoral', lw=0, zorder=0)
box1['medians'][0].set( color = 'grey', lw=2, zorder=20)
box1['medians'][1].set( color = 'grey', lw=2, zorder=20)
box1['medians'][2].set( color = 'grey', lw=2, zorder=20)
box1['medians'][3].set( color = 'grey', lw=2, zorder=20)
plt.plot([-0.5,3.5], [0, 0], '--k')      
plt.scatter(np.random.normal(loc = 0., scale = 0.08, size = 24), 
            corr_alphabeta_z[0,:], s = 50, color = 'firebrick', 
            edgecolor = 'black', zorder = 20)
plt.scatter(np.random.normal(loc = 1., scale = 0.08, size = 24), 
            corr_alphabeta_z[1,:], s = 50, color = 'crimson', 
            edgecolor = 'black', zorder = 20)    
plt.scatter(np.random.normal(loc = 2., scale = 0.08, size = 24), 
            corr_alphabeta_z[2,:], s = 50, color = 'indianred', 
            edgecolor = 'black', zorder = 20)
plt.scatter(np.random.normal(loc = 3., scale = 0.08, size = 24), 
            corr_alphabeta_z[3,:], s = 50, color = 'lightcoral', 
            edgecolor = 'black', zorder = 20)    
plt.xticks(range(4), ['alpha c1\nvs. beta c1', 'alpha c1\nvs. beta c2', 'alpha c2\nvs.beta c1', 'alpha c2\nvs. beta c2'], rotation = 0, fontsize=13)
plt.locator_params(axis='y', nbins=6)
#plt.xlim([-.45, 3.45]);
plt.ylim([-0.15,1.2])
maxY = np.nanmean(corr_alphabeta_z[3,:])
ax.annotate('***', xy=(0, maxY + 0.4), 
        ha = 'center')
ax.annotate('***', xy=(1, maxY + 0.4), 
        ha = 'center')
ax.annotate('***', xy=(2, maxY + 0.4), 
        ha = 'center')
ax.annotate('***', xy=(3, maxY + 0.4),ha = 'center')
plt.savefig(figures_dir + "Fig5B.png",dpi=800, bbox_inches='tight',transparent=True)

