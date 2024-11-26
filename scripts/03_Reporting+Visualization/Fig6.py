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

#Pupil correlations
SCP_pupil_corr_p = np.zeros((len(HLTP.subjects)))
SCP_pupil_corr_rho = np.zeros((len(HLTP.subjects)))

cluster_alphapupil_p = np.zeros((len(all_clusters),len(HLTP.subjects)))
cluster_alphapupil_rho = np.zeros((len(all_clusters),len(HLTP.subjects)))

cluster_betapupil_p = np.zeros((len(all_clusters),len(HLTP.subjects)))
cluster_betapupil_rho = np.zeros((len(all_clusters),len(HLTP.subjects)))

for s, sub in enumerate(HLTP.subjects):
    for cluster in range(len(all_clusters)):
        df = alldata_allsubs[(alldata_allsubs.subjID == sub) & (alldata_allsubs.clusterID == cluster)]
        df_clean = df.dropna() #Careful, because df now contains pupilsize column, and there is no data for s = 16, it removes s = 16 from all further analyses/plotting even if they don't involve pupil size. Stats and plots were made before this change

        SCP_pupil_corr_rho[s],SCP_pupil_corr_p[s] = spearmanr(df_clean['SCPdv'],df_clean['PupilSize'])
        cluster_alphapupil_rho[cluster,s],cluster_alphapupil_p[cluster,s] = spearmanr(df_clean['alpha'],df_clean['PupilSize'])
        cluster_betapupil_rho[cluster,s],cluster_betapupil_p[cluster,s] = spearmanr(df_clean['beta'],df_clean['PupilSize'])  

##Assessing significance of the correlations
corr_alpha_pupil_z = np.arctanh(cluster_alphapupil_rho)
corr_alpha_pupil_z = np.delete(corr_alpha_pupil_z,16,1) #Sub 16 has no pupil data
corr_beta_pupil_z = np.arctanh(cluster_betapupil_rho)
corr_beta_pupil_z = np.delete(corr_beta_pupil_z,16,1) #Sub 16 has no pupil data

#Compute whether correlations are statistically significant across subjects
pvalues_alpha_pupil = np.zeros((all_clusters.shape[0]))
Wvalues_alpha_pupil = np.zeros((all_clusters.shape[0]))
pvalues_beta_pupil = np.zeros((all_clusters.shape[0]))
Wvalues_beta_pupil = np.zeros((all_clusters.shape[0]))

#Wvalue_SCP_pupil, pvalue_SCP_pupil = wilcoxon(corr_SCP_pupil_z)
for cluster in range(len(all_clusters)):
    Wvalues_alpha_pupil[cluster],pvalues_alpha_pupil[cluster] = wilcoxon(corr_alpha_pupil_z[cluster,:])
    Wvalues_beta_pupil[cluster],pvalues_beta_pupil[cluster] = wilcoxon(corr_beta_pupil_z[cluster,:])

corr_SCP_pupil_z = np.load(data_dir + 'SCP_pupil_correlation.npy')
corr_SCP_pupil_z = np.delete(corr_SCP_pupil_z,16) #Sub 16 has no pupil data
Wvalue_SCP_pupil, pvalue_SCP_pupil = wilcoxon(corr_SCP_pupil_z)

box_style=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
colors = {'AlphaC1_high':np.array([164, 210, 255]) / 255., 
          'AlphaC2_high':np.array([4, 51, 255]) / 255., 
          'AlphaC1_low':np.array([255, 221, 140]) / 255.,
          'BetaC1_high':np.array([174, 236, 131]) / 255.,
          'BetaC2_high':np.array([53, 120, 33]) / 255., 
          'BetaC1_low':np.array([244, 170, 59]) / 255.,
          'BetaC2_low':np.array([255, 102, 102]) / 255.,
          'SCP_pupil':np.array([204, 204, 255]) / 255.}        

#Plot alpha vs. pupil
data = [corr_alpha_pupil_z[0,:],corr_alpha_pupil_z[1,:]]
fig, ax = plt.subplots(1, 1, figsize = (3, 2.5))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('outward', 10))
ax.yaxis.set_ticks_position('left')
ax.spines['bottom'].set_position(('outward', 15))
ax.xaxis.set_ticks_position('bottom')
ax.axes.xaxis.set_ticks([])
box1 = plt.boxplot(data, positions = [0,1], patch_artist = True, 
                   widths = 0.8,showfliers=False,
         boxprops=None,    showbox=None,     whis = 0, showcaps = False)
box1['boxes'][0].set( facecolor = colors['AlphaC1_high'], lw=0, zorder=0)
box1['boxes'][1].set( facecolor = colors['AlphaC2_high'], lw=0, zorder=0)
box1['medians'][0].set( color = 'grey', lw=2, zorder=20)
box1['medians'][1].set( color = 'grey', lw=2, zorder=20)
plt.plot([-0.5,1.5], [0, 0], '--k')  
#plt.plot([0], [cluster_corr_z[0,:,0]], 'o', 
         # markerfacecolor = colors['AlphaC1_low'], color = 'black', 
         # alpha = 1.);    
plt.scatter(np.random.normal(loc = 0., scale = 0.05, size = 23), 
                corr_alpha_pupil_z[0,:], s = 50, color = colors['AlphaC1_high'], 
                edgecolor = 'black', zorder = 20,clip_on=False)
plt.scatter(np.random.normal(loc = 1., scale = 0.05, size = 23), 
                corr_alpha_pupil_z[1,:], s = 50, color = colors['AlphaC2_high'], 
                edgecolor = 'black', zorder = 20,clip_on=False)
plt.xticks(range(2), ['Alpha\nCluster #1','Alpha\nCluster #2'], rotation = 0, fontsize=13)
plt.locator_params(axis='y', nbins=6)
plt.ylabel('alpha power vs.\npupil size correlation', fontsize=13)
plt.xlim([-.45, 1.45]);
plt.ylim([-0.15,0.5])
maxY = np.nanmean(corr_alpha_pupil_z[1,:])
# pvalues_alpha_pupil = 0.0233 and 0.012
ax.annotate('*', xy=(0, maxY + 0.35), 
            ha = 'center')
ax.annotate('*', xy=(1, maxY + 0.35), 
            ha = 'center')
plt.savefig(figures_dir + "Fig6A.png",dpi=800, bbox_inches='tight',transparent=True)
plt.clf()

#Plot beta vs. pupil
data = [corr_beta_pupil_z[0,:],corr_beta_pupil_z[1,:]]
fig, ax = plt.subplots(1, 1, figsize = (3, 2.5))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('outward', 10))
ax.yaxis.set_ticks_position('left')
ax.spines['bottom'].set_position(('outward', 15))
ax.xaxis.set_ticks_position('bottom')
ax.axes.xaxis.set_ticks([])
box1 = plt.boxplot(data, positions = [0,1], patch_artist = True, 
                   widths = 0.8,showfliers=False,
         boxprops=None,    showbox=None,     whis = 0, showcaps = False)
box1['boxes'][0].set( facecolor = colors['BetaC1_high'], lw=0, zorder=0)
box1['boxes'][1].set( facecolor = colors['BetaC2_high'], lw=0, zorder=0)
box1['medians'][0].set( color = 'grey', lw=2, zorder=20)
box1['medians'][1].set( color = 'grey', lw=2, zorder=20)
plt.plot([-0.5,1.5], [0, 0], '--k')  
#plt.plot([0], [cluster_corr_z[0,:,0]], 'o', 
         # markerfacecolor = colors['AlphaC1_low'], color = 'black', 
         # alpha = 1.);    
plt.scatter(np.random.normal(loc = 0., scale = 0.05, size = 23), 
                corr_beta_pupil_z[0,:], s = 50, color = colors['BetaC1_high'], 
                edgecolor = 'black', zorder = 20,clip_on=False)
plt.scatter(np.random.normal(loc = 1., scale = 0.05, size = 23), 
                corr_beta_pupil_z[1,:], s = 50, color = colors['BetaC2_high'], 
                edgecolor = 'black', zorder = 20,clip_on=False)
plt.xticks(range(2), ['Beta\nCluster #1','Beta\nCluster #2'], rotation = 0, fontsize=13)
plt.locator_params(axis='y', nbins=6)
plt.ylabel('beta power vs.\npupil size correlation', fontsize=13)
plt.xlim([-.45, 1.45]);
plt.ylim([-0.15,0.5])
maxY = np.nanmean(corr_beta_pupil_z[1,:])
# pvalues_beta_pupil = 0.00145841 and 0.00034928
ax.annotate('**', xy=(0, maxY + 0.35), 
            ha = 'center')
ax.annotate('***', xy=(1, maxY + 0.35), 
            ha = 'center')
plt.savefig(figures_dir + "Fig6B.png",dpi=800, bbox_inches='tight',transparent=True)
plt.clf()

#SCP vs. pupil
data = [corr_SCP_pupil_z]
fig, ax = plt.subplots(1, 1, figsize = (1.5, 2.5))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('outward', 10))
ax.yaxis.set_ticks_position('left')
ax.spines['bottom'].set_position(('outward', 15))
ax.xaxis.set_ticks_position('bottom')
ax.axes.xaxis.set_ticks([])
box1 = plt.boxplot(data, positions = [0], patch_artist = True, 
                   widths = 0.8,showfliers=False,
         boxprops=None,    showbox=None,     whis = 0, showcaps = False)
box1['boxes'][0].set( facecolor = [0.9,0.9,0.9], lw=0, zorder=0)
box1['medians'][0].set( color = 'grey', lw=2, zorder=20)
plt.plot([-0.5,0.5], [0, 0], '--k')    
plt.scatter(np.random.normal(loc = 0., scale = 0.05, size = 23), 
                corr_SCP_pupil_z, s = 50, color = [0.9,0.9,0.9], 
                edgecolor = 'black', zorder = 20,clip_on=False)
#plt.xticks(range(1), ['Beta\nCluster #1','Beta\nCluster #2'], rotation = 0, fontsize=13)
plt.locator_params(axis='y', nbins=6)
plt.ylabel('SCP d.v. vs.\npupil size correlation', fontsize=13)
plt.xlim([-.45, 0.45]);
plt.ylim([-0.15,0.5])
maxY = np.nanmean(corr_SCP_pupil_z)
ax.annotate('**', xy=(0, maxY + 0.35), 
            ha = 'center')
plt.savefig(figures_dir + "Fig6C.png",dpi=800, bbox_inches='tight',transparent=True)
plt.clf()
 