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

nSubs = 11
n_prestim_timepoints = 4
nSensors = 273

corr_alpha_betac1 = pd.read_csv(data_dir + 'alpha_betac1_corr.csv')
corr_alpha_betac1 = corr_alpha_betac1.values

corr_alpha_betac2 = pd.read_csv(data_dir + 'alpha_betac2_corr.csv')
corr_alpha_betac2 = corr_alpha_betac2.values

#### PLOT CORRELATIONS BETWEEN SCP, ALPHA, BETA for ALL TRIALS ####
# Scatterplots
# Alpha/beta vs. SCP
x = np.arange(nSubs)
ys = [i+x+(i*x)**2 for i in range(nSubs)]
import matplotlib.cm as cm
colors = cm.Paired(np.linspace(0, 0.9, len(ys)))

# Barplots
box_style=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
colors = {'AlphaC1_high':np.array([164, 210, 255]) / 255., 
      'AlphaC2_high':np.array([4, 51, 255]) / 255., 
      'AlphaC1_low':np.array([255, 221, 140]) / 255.,
      'BetaC1_high':np.array([174, 236, 131]) / 255.,
      'BetaC2_high':np.array([53, 120, 33]) / 255., 
      'BetaC1_low':np.array([244, 170, 59]) / 255.,
      'BetaC2_low':np.array([255, 102, 102]) / 255.}        

# All clusters alpha vs beta
data = [corr_alpha_betac1,corr_alpha_betac2]
data = [arr.flatten() for arr in data]
fig, ax = plt.subplots(1, 1, figsize = (3.5, 2.5))
ax.spines['left'].set_position(('outward', 10))
ax.yaxis.set_ticks_position('left')
ax.spines['bottom'].set_position(('outward', 15))
ax.xaxis.set_ticks_position('bottom')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.axes.xaxis.set_ticks([])
box1 = plt.boxplot(data, positions = [0,1], patch_artist = True, 
                   widths = 0.8,showfliers=False,
         boxprops=None,    showbox=None,     whis = 0, showcaps = False)
box1['boxes'][0].set( facecolor = 'palevioletred', lw=0, zorder=0)
box1['boxes'][1].set( facecolor = 'lightpink', lw=0, zorder=0)
box1['medians'][0].set( color = 'grey', lw=2, zorder=20)
box1['medians'][1].set( color = 'grey', lw=2, zorder=20)
plt.plot([-0.5,1.5], [0, 0], '--k')  

plt.scatter(np.random.normal(loc = 0., scale = 0.05, size = 11), 
                corr_alpha_betac1, s = 50, color = 'palevioletred', 
                edgecolor = 'black', zorder = 20)
plt.scatter(np.random.normal(loc = 1., scale = 0.05, size = 11), 
                corr_alpha_betac2, s = 50, color = 'lightpink', 
                edgecolor = 'black', zorder = 20)
plt.xticks(range(2), ['alpha c1\nvs. beta c1', 'alpha c1\nvs. beta c2'], rotation = 0, fontsize=13)
plt.locator_params(axis='y', nbins=6)
#ax.set_xlim([0., 2.]);
plt.xlim([-.45, 1.45]);
plt.ylim([-0.2,0.8])
maxY = np.nanmean(corr_alpha_betac1)
ax.annotate('***', xy=(0, maxY + 0.55), 
            ha = 'center')
ax.annotate('***', xy=(1, maxY + 0.55), 
            ha = 'center')
plt.savefig(figures_dir + "Fig5A.png",dpi=800, bbox_inches='tight',transparent=True)
plt.clf()