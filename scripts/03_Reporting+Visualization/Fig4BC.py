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
data_dir = '/isilon/LFMI/VMdrive/Lua/MEG_alpha_1-f_SCP_PLoSBiol_2024/data/HL/'
figures_dir = '/isilon/LFMI/VMdrive/Lua/MEG_alpha_1-f_SCP_PLoSBiol_2024/figures/'
supp_dir = '/isilon/LFMI/VMdrive/Lua/MEG_alpha_1-f_SCP_PLoSBiol_2024/scripts/supporting_files_toolboxes/HL/'

nSubs = 11
n_prestim_timepoints = 4
nSensors = 273

alldata_allsubs = pd.read_pickle(data_dir + "alldata_allsubjects_cluster.pkl")

#Convert alpha and beta values to femtoTeslas
alldata_allsubs['alpha'] = (np.sqrt(alldata_allsubs['alpha'])*10e-15)**2
alldata_allsubs['beta'] = (np.sqrt(alldata_allsubs['beta'])*10e-15)**2

beta_auroc = np.load(data_dir + 'beta_auroc.npy')
SCP_auroc = np.load(data_dir + 'SCP_auroc.npy')
beta_noSCP_auroc = np.load(data_dir + 'beta_noSCP_auroc.npy')
SCP_nobeta_auroc = np.load(data_dir + 'SCP_nobeta_auroc.npy')

#Make bar plots comparing AUROCs across subjects
box_style=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
box_style2 = dict(boxstyle='round',facecolor='#F51445',alpha=0.5)

names1 = ['Fig4B_1','Fig4C_1']
for cluster in range(2,4):
    W, p = wilcoxon(SCP_auroc[:,cluster],SCP_nobeta_auroc[:,cluster])
    W1,p1 = wilcoxon(SCP_auroc[:,cluster]-0.5,alternative='greater')
    W2,p2 = wilcoxon(SCP_nobeta_auroc[:,cluster]-0.5,alternative='greater')
    fig, ax = plt.subplots(1, 1, figsize = (3, 2.5))
    data = [SCP_auroc[:,cluster],SCP_nobeta_auroc[:,cluster]]
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
    box1['boxes'][0].set( facecolor = '#7030A0', lw=0, zorder=0)
    box1['boxes'][1].set( facecolor = '#D349FF', lw=0, zorder=0)
    box1['medians'][0].set( color = 'grey', lw=2, zorder=20)
    box1['medians'][1].set( color = 'grey', lw=2, zorder=20)
    plt.plot([-0.5,1.5], [0.5, 0.5], '--k')  
    plt.scatter(np.random.normal(loc = 0., scale = 0.08, size = 24), 
                SCP_auroc[:,cluster], s = 50, color = '#7030A0', 
                edgecolor = 'black', zorder = 20)
    plt.scatter(np.random.normal(loc = 1., scale = 0.08, size = 24), 
                SCP_nobeta_auroc[:,cluster], s = 50, color = '#D349FF', 
                edgecolor = 'black', zorder = 20)
    plt.xticks(range(2), ['SCP d.v.', 'SCP d.v.\nresiduals'], rotation = 0, fontsize=13)
    plt.locator_params(axis='y', nbins=6)
    plt.ylabel('AUROC')
    plt.xlim([-.45, 1.45]);
    plt.ylim([0.4,0.9])
    maxY = np.nanmean(beta_auroc[:,cluster])
    if (p1 < 0.05) & (p1 >= 0.01):
        ax.annotate('*', xy=(0, maxY + 0.2), ha = 'center')
    elif (p1 < 0.01) & (p1 >= 0.001):
        ax.annotate('**', xy=(0, maxY + 0.2), ha = 'center')
    elif p1 < 0.001:
        ax.annotate('***', xy=(0, maxY + 0.2), ha = 'center')
    else:
        ax.annotate('n.s.', xy=(0, maxY + 0.2), ha = 'center')
    if (p2 < 0.05) & (p2 >= 0.01):
        ax.annotate('*', xy=(1, maxY + 0.2), ha = 'center')
    elif (p2 < 0.01) & (p2 >= 0.001):
        ax.annotate('**', xy=(1, maxY + 0.2), ha = 'center')
    elif p2 < 0.001:
        ax.annotate('***', xy=(1, maxY + 0.2), ha = 'center')
    else:
        ax.annotate('n.s.', xy=(1, maxY + 0.2), ha = 'center')
    ax.annotate('p = ' + str(round(p,2)), xy=(0.5, maxY + 0.35), zorder=10, ha = 'center')
    props = {'connectionstyle':'bar','arrowstyle':'-',\
                     'shrinkA':15,'shrinkB':15}
    ax.annotate('', xy=(0, maxY+0.24), xytext=(1, maxY+0.24), arrowprops=props)
    plt.savefig(figures_dir + names1[cluster-2] + ".png",dpi=800, bbox_inches='tight',transparent=True)
    plt.clf()
    
names2 = ['Fig4B_2','Fig4C_2']
for cluster in range(2,4):
    W, p = wilcoxon(beta_auroc[:,cluster],beta_noSCP_auroc[:,cluster])
    W1,p1 = wilcoxon(beta_auroc[:,cluster]-0.5,alternative='greater')
    W2,p2 = wilcoxon(beta_noSCP_auroc[:,cluster]-0.5,alternative='greater')
    fig, ax = plt.subplots(1, 1, figsize = (3, 2.5))
    data = [beta_auroc[:,cluster],beta_noSCP_auroc[:,cluster]]
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.spines['bottom'].set_position(('outward', 15))
    ax.xaxis.set_ticks_position('bottom')
    ax.axes.xaxis.set_ticks([])
    box1 = plt.boxplot(data, positions = [0,1], patch_artist = True, 
                       widths = 0.8,showfliers=False,
             boxprops=None,    showbox=None,     whis = 0, showcaps = False)
    box1['boxes'][0].set( facecolor = '#FF32D8', lw=0, zorder=0)
    box1['boxes'][1].set( facecolor = '#F8C2F0', lw=0, zorder=0)
    box1['medians'][0].set( color = 'grey', lw=2, zorder=20)
    box1['medians'][1].set( color = 'grey', lw=2, zorder=20)
    plt.plot([-0.5,1.5], [0.5, 0.5], '--k')  
    plt.scatter(np.random.normal(loc = 0., scale = 0.08, size = 24), 
                beta_auroc[:,cluster], s = 50, color = '#FF32D8', 
                edgecolor = 'black', zorder = 20)
    plt.scatter(np.random.normal(loc = 1., scale = 0.08, size = 24), 
                beta_noSCP_auroc[:,cluster], s = 50, color = '#F8C2F0', 
                edgecolor = 'black', zorder = 20)    
    plt.xticks(range(2), ['beta power', 'beta power\nresiduals'], rotation = 0, fontsize=13)
    plt.locator_params(axis='y', nbins=6)
    #ax.set_xlim([0., 2.]);
    plt.ylabel('AUROC')
    plt.xlim([-.45, 1.45]);
    plt.ylim([0.4,0.9])
    maxY = np.nanmean(beta_auroc[:,cluster])
    if (p1 < 0.05) & (p1 >= 0.01):
        ax.annotate('*', xy=(0, maxY + 0.2), ha = 'center')
    elif (p1 < 0.01) & (p1 >= 0.001):
        ax.annotate('**', xy=(0, maxY + 0.2), ha = 'center')
    elif p1 < 0.001:
        ax.annotate('***', xy=(0, maxY + 0.2), ha = 'center')
    else:
        ax.annotate('n.s.', xy=(0, maxY + 0.2), ha = 'center')
    if (p2 < 0.05) & (p2 >= 0.01):
        ax.annotate('*', xy=(1, maxY + 0.2), ha = 'center')
    elif (p2 < 0.01) & (p2 >= 0.001):
        ax.annotate('**', xy=(1, maxY + 0.2), ha = 'center')
    elif p2 < 0.001:
        ax.annotate('***', xy=(1, maxY + 0.2), ha = 'center')
    else:
        ax.annotate('n.s.', xy=(1, maxY + 0.2), ha = 'center')
    ax.annotate('p = ' + str(round(p,2)), xy=(0.5, maxY + 0.35), zorder=10, ha = 'center')
    props = {'connectionstyle':'bar','arrowstyle':'-',\
                     'shrinkA':15,'shrinkB':15}
    ax.annotate('', xy=(0, maxY+0.24), xytext=(1, maxY+0.24), arrowprops=props)
    plt.savefig(figures_dir + names2[cluster-2] + ".png",dpi=800, bbox_inches='tight',transparent=True)
    plt.clf()