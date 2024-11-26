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

#Read in sensor names
with open(supp_dir + 'CTF275labels.txt') as f:
    sensor_names = f.readlines()
sensor_names = np.array(sensor_names)
sensor_names = np.delete(sensor_names,HLTP.bad_ch)

alldata_allsubs = pd.read_pickle(data_dir + "alldata_allsubjects_cluster_withpupil.pkl")
allSCP_pupildata = pd.read_pickle(data_dir + "/allpupil_SCP_data.pkl")
from scipy.special import logit
allSCP_pupildata['pred_prob'] = logit(allSCP_pupildata['pred_prob'])
allSCP_pupildata.replace([np.inf, -np.inf], np.nan, inplace=True)

# Also fill out the correlation between SCPdv and alpha or beta power
cluster_corr_p = np.zeros((len(all_clusters),len(HLTP.subjects),2)) #all clusters by subjects by (alphaDV,betaDV)
cluster_corr_rho = np.zeros((len(all_clusters),len(HLTP.subjects),2))

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
        
#### PLOT CORRELATIONS BETWEEN SCP, ALPHA, BETA for ALL TRIALS ####
# Scatterplots
# Alpha/beta vs. SCP
x = np.arange(len(HLTP.subjects))
ys = [i+x+(i*x)**2 for i in range(len(HLTP.subjects))]
import matplotlib.cm as cm
colors = cm.Paired(np.linspace(0, 0.9, len(ys)))

names = ['Fig2F_1_scatter','Fig2F_2_scatter','Fig3D_1_scatter','Fig3D_2_scatter']
for cluster in range(len(all_clusters)):
    fig, ax = plt.subplots(1, 1, figsize = (4, 2))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    maxXlim = 1
    x_extended = np.linspace(0,maxXlim,100)
    df_all = pd.DataFrame(dtype=float)
    for s,sub in enumerate(HLTP.subjects):
        df = alldata_allsubs[(alldata_allsubs.subjID == sub) & (alldata_allsubs.clusterID == cluster)]
        df_clean = df.dropna()
        df_clean[['alpha_norm','beta_norm']] = (df_clean[['alpha','beta']] - df_clean[['alpha','beta']].min()) / (df_clean[['alpha','beta']].max() - df_clean[['alpha','beta']].min())
        df_all = pd.concat([df_all,df_clean[['alpha_norm','beta_norm','SCPdv']]])
        if cluster < 2: #Alpha clusters
            plt.scatter(df_clean['alpha_norm'],df_clean['SCPdv'],alpha=0.7,linewidth=0.05,color=colors[s],s=20)
            plt.xlabel("normalized alpha power",fontsize=13)
            name = "alpha_vs_SCP_scatter_realtrials_c" + str(cluster+1)
            z = np.polyfit(df_all['alpha_norm'],df_all['SCPdv'],1)
        else: #Beta clusters
            plt.scatter(df_clean['beta_norm'],df_clean['SCPdv'],alpha=0.7,linewidth=0.05,color=colors[s],s=20)
            plt.xlabel("normalized beta power",fontsize=13)
            name = "beta_vs_SCP_scatter_realtrials_c" + str(cluster-1)
            z = np.polyfit(df_all['beta_norm'],df_all['SCPdv'],1)
    plt.xlim(0,maxXlim)
    p = np.polyval(z,x_extended)
    plt.plot(x_extended,p,color='black',linewidth=2)
    plt.xticks(fontsize=12)
    plt.ylabel("SCP d.v.",fontsize=13)
    plt.yticks(fontsize=12)
    plt.savefig(figures_dir + names[cluster] + ".png",dpi=800, bbox_inches='tight',transparent=True)
    plt.clf()

box_style=dict(boxstyle='round', facecolor='wheat', alpha=0.5)

colors = {'AlphaC1_high':np.array([164, 210, 255]) / 255., 
          'AlphaC2_high':np.array([4, 51, 255]) / 255., 
          'AlphaC1_low':np.array([255, 221, 140]) / 255.,
          'BetaC1_high':np.array([174, 236, 131]) / 255.,
          'BetaC2_high':np.array([53, 120, 33]) / 255., 
          'BetaC1_low':np.array([244, 170, 59]) / 255.,
          'BetaC2_low':np.array([255, 102, 102]) / 255.,
          'SCP_pupil':np.array([204, 204, 255]) / 255.}        

# Alpha clusters
data = [cluster_corr_z[0,:,0],cluster_corr_z[1,:,0]]
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
plt.plot([0], [cluster_corr_z[0,:,0]], 'o', 
          markerfacecolor = colors['AlphaC1_high'], color = 'black', 
          alpha = 1.);    
plt.plot([1], [cluster_corr_z[1,:,0]], 'o',
          markerfacecolor = colors['AlphaC2_high'], color = 'black', alpha = 1.);      
plt.scatter(np.random.normal(loc = 0., scale = 0.08, size = 24), 
                cluster_corr_z[0,:,0], s = 50, color = colors['AlphaC1_high'], 
                edgecolor = 'black', zorder = 20)
plt.scatter(np.random.normal(loc = 1., scale = 0.08, size = 24), 
                cluster_corr_z[1,:,0], s = 50, color = colors['AlphaC2_high'], 
                edgecolor = 'black', zorder = 20)    
plt.xticks(range(2), ['Cluster #1', 'Cluster #2'], rotation = 0, fontsize=13)
plt.locator_params(axis='y', nbins=6)
ax.set_xlim([0., 1.]);
plt.ylabel('alpha power vs.\nSCP d.v. correlation', fontsize=13)
plt.xlim([-.45, 1.45]);
plt.ylim([-0.15,0.3])
maxY = np.nanmean(cluster_corr_z[1,:,0])
ax.annotate('n.s.', xy=(0, maxY + 0.25), 
            ha = 'center')
ax.annotate('n.s.', xy=(1, maxY + 0.25), 
            ha = 'center')
plt.savefig(figures_dir + "Fig2F_box.png",dpi=800, bbox_inches='tight',transparent=True)
plt.clf()
    
# Beta clusters
data = [cluster_corr_z[2,:,1],cluster_corr_z[3,:,1]]
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
# plt.plot([0], [cluster_corr_z[2,:,1]], 'o', 
#          markerfacecolor = colors['BetaC1_high'], color = 'black', 
#          alpha = 1.);    
# plt.plot([1], [cluster_corr_z[3,:,1]], 'o',
#          markerfacecolor = colors['BetaC2_high'], color = 'black', alpha = 1.);        
plt.scatter(np.random.normal(loc = 0., scale = 0.08, size = 24), 
                cluster_corr_z[2,:,1], s = 50, color = colors['BetaC1_high'], 
                edgecolor = 'black', zorder = 20)
plt.scatter(np.random.normal(loc = 1., scale = 0.08, size = 24), 
                cluster_corr_z[3,:,1], s = 50, color = colors['BetaC2_high'], 
                edgecolor = 'black', zorder = 20)      
plt.xticks(range(2), ['Cluster #1', 'Cluster #2'], rotation = 0, fontsize=13)
plt.yticks(fontsize=13)
plt.locator_params(axis='y', nbins=6)
ax.set_xlim([0., 1.]);
plt.ylabel('beta power vs.\nSCP d.v. correlation',fontsize=14)
plt.xlim([-.45, 1.45]);
plt.ylim([-0.15,0.5])
maxY = np.nanmean(cluster_corr_z[3,:,1])
ax.annotate('*', xy=(0, maxY + 0.35), 
            ha = 'center')
ax.annotate('**', xy=(1, maxY + 0.35), 
            ha = 'center')
plt.savefig(figures_dir + "Fig3D_box.png",dpi=800, bbox_inches='tight',transparent=True)
plt.clf()