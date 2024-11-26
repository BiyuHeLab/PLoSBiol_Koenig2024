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

alldata_allsubs = pd.read_pickle(data_dir + "alldata_allsubjects_cluster.pkl")

#Convert alpha and beta values to femtoTeslas
alldata_allsubs['alpha'] = (np.sqrt(alldata_allsubs['alpha'])*10e-15)**2
alldata_allsubs['beta'] = (np.sqrt(alldata_allsubs['beta'])*10e-15)**2

alpha_c1 = pd.read_csv(data_dir + 'alpha_SCP_corr_c1.csv')
alpha_c1 = alpha_c1.values

beta_c1 = pd.read_csv(data_dir + 'beta_SCP_corr_c1.csv')
beta_c1 = beta_c1.values

beta_c2 = pd.read_csv(data_dir + 'beta_SCP_corr_c2.csv')
beta_c2 = beta_c2.values

#### PLOT CORRELATIONS BETWEEN SCP, ALPHA, BETA for ALL TRIALS ####
# Scatterplots
# Alpha/beta vs. SCP
x = np.arange(nSubs)
ys = [i+x+(i*x)**2 for i in range(nSubs)]
import matplotlib.cm as cm
colors = cm.Paired(np.linspace(0, 0.9, len(ys)))

names = ['Fig2C_scatter','Fig3B_1_scatter','Fig3B_2_scatter']
for cluster in range(len(all_clusters)):
    fig, ax = plt.subplots(1, 1, figsize = (4, 2))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    maxXlim = 1
    x_extended = np.linspace(0,maxXlim,100)
    df_all = pd.DataFrame(dtype=float)
    for s in range(nSubs):
        df = alldata_allsubs[(alldata_allsubs.subjID == s) & (alldata_allsubs.clusterID == cluster)]
        df_clean = df.dropna()
        df_clean[['alpha_norm','beta_norm']] = (df_clean[['alpha','beta']] - df_clean[['alpha','beta']].min()) / (df_clean[['alpha','beta']].max() - df_clean[['alpha','beta']].min())
        df_all = pd.concat([df_all,df_clean[['alpha_norm','beta_norm','SCPdv']]])
        if cluster < 1: #Alpha clusters
            plt.scatter(df_clean['alpha_norm'],df_clean['SCPdv'],alpha=0.5)
            plt.xlabel("normalized alpha power",fontsize=13)
            z = np.polyfit(df_all['alpha_norm'],df_all['SCPdv'],1)
        else: #Beta clusters
            plt.scatter(df_clean['beta_norm'],df_clean['SCPdv'],alpha=0.5)
            plt.xlabel("normalized beta power",fontsize=13)
            z = np.polyfit(df_all['beta_norm'],df_all['SCPdv'],1)
    plt.xlim(0,maxXlim)
    p = np.polyval(z,x_extended)
    plt.plot(x_extended,p,color='black',linewidth=2)
    plt.xticks(fontsize=12)
    plt.ylabel("SCP d.v.",fontsize=13)
    plt.yticks(fontsize=12)
    plt.savefig(figures_dir + names[cluster] + ".png",dpi=800, bbox_inches='tight',transparent=True)
    plt.clf()

# Barplots
box_style=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
colors = {'AlphaC1_high':np.array([164, 210, 255]) / 255., 
      'AlphaC2_high':np.array([4, 51, 255]) / 255., 
      'AlphaC1_low':np.array([255, 221, 140]) / 255.,
      'BetaC1_high':np.array([174, 236, 131]) / 255.,
      'BetaC2_high':np.array([53, 120, 33]) / 255., 
      'BetaC1_low':np.array([244, 170, 59]) / 255.,
      'BetaC2_low':np.array([255, 102, 102]) / 255.}        

# Alpha cluster
data = [alpha_c1]
data = [arr.flatten() for arr in data]
fig, ax = plt.subplots(1, 1, figsize = (1.5, 2.5))
ax.spines['left'].set_position(('outward', 10))
ax.yaxis.set_ticks_position('left')
ax.spines['bottom'].set_position(('outward', 15))
ax.xaxis.set_ticks_position('bottom')
ax.axes.xaxis.set_ticks([])
box1 = plt.boxplot(data, positions = [0], patch_artist = True, 
                   widths = 0.8,showfliers=False,
         boxprops=None,    showbox=None,     whis = 0, showcaps = False)
box1['boxes'][0].set( facecolor = colors['AlphaC1_low'], lw=0, zorder=0)
box1['medians'][0].set( color = 'grey', lw=2, zorder=20)
plt.plot([-0.5,0.5], [0, 0], '--k') 
plt.scatter(np.random.normal(loc = 0., scale = 0.05, size = 11), 
                alpha_c1, s = 50, color = colors['AlphaC1_low'], 
                edgecolor = 'black', zorder = 20)
plt.xticks(range(1), ['Cluster #1'], rotation = 0, fontsize=13)
plt.locator_params(axis='y', nbins=6)
plt.ylabel('alpha power vs.\nSCP d.v. correlation')
plt.xlim([-.45, 0.45]);
plt.ylim([-0.15,0.3])
maxY = np.nanmean(alpha_c1)
ax.annotate('n.s.', xy=(0, maxY + 0.25), 
            ha = 'center')
plt.savefig(figures_dir + "Fig2C_box.png",dpi=800, bbox_inches='tight',transparent=True)
plt.clf()
    
# Beta clusters
data = [beta_c1,beta_c2]
data = [arr.flatten() for arr in data]
fig, ax = plt.subplots(1, 1, figsize = (3, 2.5))
ax.spines['left'].set_position(('outward', 10))
ax.yaxis.set_ticks_position('left')
ax.spines['bottom'].set_position(('outward', 15))
ax.xaxis.set_ticks_position('bottom')
ax.axes.xaxis.set_ticks([])
box1 = plt.boxplot(data, positions = [0,1], patch_artist = True, 
                   widths = 0.8,showfliers=False,
         boxprops=None,    showbox=None,     whis = 0, showcaps = False)
box1['boxes'][0].set( facecolor = colors['BetaC1_low'], lw=0, zorder=0)
box1['boxes'][1].set( facecolor = colors['BetaC2_low'], lw=0, zorder=0)
box1['medians'][0].set( color = 'grey', lw=2, zorder=20)
box1['medians'][1].set( color = 'grey', lw=2, zorder=20)
plt.plot([-0.5,1.5], [0, 0], '--k')  
plt.scatter(np.random.normal(loc = 0., scale = 0.05, size = 11), 
                beta_c1, s = 50, color = colors['BetaC1_low'], 
                edgecolor = 'black', zorder = 20)
plt.scatter(np.random.normal(loc = 1., scale = 0.05, size = 11), 
                beta_c2, s = 50, color = colors['BetaC2_low'], 
                edgecolor = 'black', zorder = 20)
plt.xticks(range(2), ['Cluster #1', 'Cluster #2'], rotation = 0, fontsize=13)
plt.locator_params(axis='y', nbins=6)
ax.set_xlim([0., 1.]);
plt.ylabel('beta power vs.\nSCP d.v. correlation')
plt.xlim([-.45, 1.45]);
plt.ylim([-0.2,0.4])
maxY = np.nanmean(beta_c1)
ax.annotate('n.s.', xy=(0, maxY + 0.35), 
            ha = 'center')
ax.annotate('n.s.', xy=(1, maxY + 0.35), 
            ha = 'center')
plt.savefig(figures_dir + "Fig3B_box.png",dpi=800, bbox_inches='tight',transparent=True)
plt.clf()