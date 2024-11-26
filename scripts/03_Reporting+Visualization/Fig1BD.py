#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 16:05:50 2023

@author: koenil04
"""

from matplotlib import rcParams
import matplotlib.pyplot as plt
from mne import viz
import numpy as np
import pandas as pd
import pickle
import os
from scipy.io import loadmat, savemat

os.chdir('/isilon/LFMI/VMdrive/Lua/MEG_alpha_1-f_SCP_PLoSBiol_2024/scripts')
data_dir = '/isilon/LFMI/VMdrive/Lua/MEG_alpha_1-f_SCP_PLoSBiol_2024/data/behav/'
figures_dir = '/isilon/LFMI/VMdrive/Lua/MEG_alpha_1-f_SCP_PLoSBiol_2024/figures/'

import HLTP
from scipy import stats

colors = {'Rec':np.array([191, 152, 218]) / 255.,
          'RecDark':np.array([117, 43, 168]) / 255.,
          'Incorr':np.array([150, 194, 221]) /255.,
          'IncorrDark':np.array([75, 106, 221]) / 255.,
          'CorrDark':np.array([234, 56, 148]) / 255.,
          'Corr':np.array([234, 165, 224]) / 255.    }

fig_width = 7  # width in inches
fig_height = 4.2  # height in inches
fig_size =  [fig_width,fig_height]
params = {    
          'axes.spines.right': False,
          'axes.spines.top': False,
          
          'figure.figsize': fig_size,
          
          'ytick.major.size' : 8,      # major tick size in points
          'xtick.major.size' : 8,    # major tick size in points
              
          'lines.markersize': 6.0,
          # font size
          'axes.labelsize': 14,
          'axes.titlesize': 14,     
          'xtick.labelsize': 12,
          'ytick.labelsize': 12,
          'font.size': 12,

          # linewidth
          'axes.linewidth' : 1.3,
          'patch.linewidth': 1.3,
          
          'ytick.major.width': 1.3,
          'xtick.major.width': 1.3,
          'savefig.dpi' : 800
          }
rcParams.update(params)
rcParams['font.sans-serif'] = 'Helvetica'


def print_percentseen_Podvalny():
    ''' Percent Recognized in real and scrambled trials'''
    percent_seen_real = HLTP.load(data_dir + 'Podvalny_percent_seen_real.p')
    percent_seen_scra = pd.read_pickle(data_dir + 'Podvalny_percent_seen_scra.p')
    
    data= [percent_seen_real, percent_seen_scra]
    fig, ax = plt.subplots(1, 1, figsize = (1.4, 2.5))
    
    ax.spines['left'].set_position(('outward', 10))
    ax.yaxis.set_ticks_position('left')
    ax.spines['bottom'].set_position(('outward', 15))
    ax.xaxis.set_ticks_position('bottom')
    box1 = plt.boxplot(data, positions = [0,1], patch_artist = True, 
                       widths = 0.8,showfliers=False,
             boxprops=None,    showbox=None,     whis = 0, showcaps = False)
    
    box1['boxes'][0].set( facecolor = colors['Rec'], lw=0, zorder=0)
    box1['boxes'][1].set( facecolor = [.9,.9,.9], lw=0, zorder=0)
    
    box1['medians'][0].set( color = colors['RecDark'], lw=2, zorder=20)
    box1['medians'][1].set( color =  [.4,.4,.4], lw=2, zorder=20)

    plt.plot([-0.5,1.5], [50, 50], '--k')
    
    plt.plot([0, 1], [percent_seen_real, percent_seen_scra], 
             color = [.5, .5, .5], lw = 0.5);
    plt.scatter(np.random.normal(loc = 0., scale = 0.08, size = 24), 
                percent_seen_real, s = 50, color = colors['Rec'],
                edgecolor = colors['RecDark'], zorder = 20,clip_on=False)
    plt.scatter(np.random.normal(loc = 1., scale = 0.08, size = 24), 
                percent_seen_scra, s = 50, color = [.9, .9, .9], 
                edgecolor = colors['RecDark'], zorder = 20,clip_on=False) 
    plt.xticks(range(2), ['Real', 'Scram'], rotation = 0, fontsize=13)
    plt.locator_params(axis='y', nbins=6)
    #ax.set_xticks( (-0., 1.))
    ax.set_xlim([0., 1.]);
    plt.ylabel('% Recognized')
    plt.ylim([-0.01, 100]);plt.xlim([-.45, 1.45]);
 
    props = {'connectionstyle':'bar','arrowstyle':'-',\
                     'shrinkA':5,'shrinkB':5}
    mx = percent_seen_real.max() + 5
    ax.annotate('***', xy=(0.5, mx + 10), zorder=10, ha = 'center')
    ax.annotate('', xy=(0, mx), xytext=(1, mx), arrowprops=props)
    
    fig.savefig(figures_dir + 'Fig1D_1.png', 
                bbox_inches = 'tight', transparent=True)

def print_percentcorrect_Podvalny():
    '''Percent correct category reports by trial type
    real/scrambled and recognized/unrecognized'''
    correct_R_real = HLTP.load(data_dir + 'Podvalny_correct_seen.p')
    correct_U_real = HLTP.load(data_dir + 'Podvalny_correct_unseen.p')        
    correct_R_scra = HLTP.load(data_dir + 'Podvalny_correct_seen_scra.p')
    correct_U_scra = HLTP.load(data_dir + 'Podvalny_correct_unseen_scra.p')   
    
    xdata = [0, 1, 2.5, 3.5]
    data = [correct_R_real, correct_U_real, 
         correct_R_scra, correct_U_scra]
    fig, ax = plt.subplots(1, 1,figsize = (2.5, 2.5))
    ax.spines['left'].set_position(('outward', 10))
    ax.yaxis.set_ticks_position('left')
    ax.spines['bottom'].set_position(('outward', 15))
    ax.xaxis.set_ticks_position('bottom')
    box1 = plt.boxplot(data, positions = xdata, patch_artist = True, widths = 0.8,
             showfliers = False, boxprops = None, showbox = None, 
             whis = 0, showcaps = False)
    box1['boxes'][0].set( facecolor = colors['Corr'], lw=0, zorder=0)    
    box1['boxes'][1].set( facecolor = colors['Incorr'], lw=0, zorder=0)    
    box1['boxes'][2].set( facecolor = [.9,.9,.9], lw=0, zorder=0)    
    box1['boxes'][3].set( facecolor = [.9,.9,.9], lw=0, zorder=0)    

    box1['medians'][0].set( color = colors['CorrDark'], lw=2, zorder=20)
    box1['medians'][1].set( color = colors['IncorrDark'], lw=2, zorder=20)
    box1['medians'][2].set( color = colors['CorrDark'], lw=2, zorder=20)
    box1['medians'][3].set( color = colors['IncorrDark'], lw=2, zorder=20)

    plt.plot(xdata[:2], [correct_R_real, correct_U_real],
             color = [.5,.5,.5], lw = 0.5);
    plt.plot(xdata[2:], [correct_R_scra, correct_U_scra],
             color = [.5,.5,.5], lw = 0.5);       
    plt.plot([-0.5,4], [25, 25], '--k')

    plt.scatter(np.random.normal(loc = 0., scale = 0.08, size = 24), 
                correct_R_real, s = 50, color = colors['Corr'],
                edgecolor = colors['CorrDark'], zorder = 20,clip_on=False)
    plt.scatter(np.random.normal(loc = 1., scale = 0.08, size = 24), 
                correct_U_real, s = 50, color = colors['Incorr'],
                edgecolor = colors['IncorrDark'], zorder = 20)
    plt.scatter(np.random.normal(loc = 2.5, scale = 0.08, size = correct_R_scra.size), 
                correct_R_scra, s = 50, color = [.9,.9,.9],
                edgecolor = colors['CorrDark'], zorder = 20)
    plt.scatter(np.random.normal(loc = 3.5, scale = 0.08, size = correct_U_scra.size), 
                correct_U_scra, s = 50, color = [.9,.9,.9],
                edgecolor = colors['IncorrDark'], zorder = 20)

    plt.xticks(xdata, ['Y', 'N', 'Y', 'N'],fontsize = 15)
    plt.yticks(fontsize=15)
    plt.ylabel('% Correct',fontsize=15)
    plt.ylim([0, 115]); plt.xlim([-0.5, 4.]);
    plt.locator_params(axis='y', nbins=5)

    props = {'connectionstyle':'bar','arrowstyle':'-',\
                     'shrinkA':4,'shrinkB':4}
    mx = correct_R_real.max() + 5
    ax.annotate('***', xy=(0.5, mx +7), zorder=10, ha = 'center')
    ax.annotate('', xy=(0, mx), xytext=(1, mx), arrowprops=props)
    
    ax.annotate('**', xy=(3., mx +7), zorder=10, ha = 'center')
    ax.annotate('', xy=(2.5, mx), xytext=(3.5, mx), arrowprops=props)
    fig.savefig(figures_dir + 'Fig1D_2.png', dpi=800, 
                transparent=True, bbox_inches = 'tight')
    
def print_percentseen_correct_Baria():
    ''' Percent Recognized in real and scrambled trials'''
    percent_seen_real = loadmat(data_dir + 'Baria_percentseen.mat')['p_seen'][0,:]
    percent_seen_catch = loadmat(data_dir + 'Baria_percentseen_catch.mat')['Hitrate_catch'][0,:]
    
    data= [percent_seen_real,percent_seen_catch]
    
    fig, ax = plt.subplots(1, 1, figsize = (1.4, 2.5))
    
    ax.spines['left'].set_position(('outward', 10))
    ax.yaxis.set_ticks_position('left')
    ax.spines['bottom'].set_position(('outward', 15))
    ax.xaxis.set_ticks_position('bottom')
    box1 = plt.boxplot(data, positions = [0,1], patch_artist = True, 
                       widths = 0.8,showfliers=False,
             boxprops=None,    showbox=None,     whis = 0, showcaps = False)
    
    box1['boxes'][0].set( facecolor = colors['Rec'], lw=0, zorder=0)
    box1['boxes'][1].set( facecolor = [.9,.9,.9], lw=0, zorder=0)
    
    box1['medians'][0].set( color = colors['RecDark'], lw=2, zorder=20)
    box1['medians'][1].set( color =  [.4,.4,.4], lw=2, zorder=20)

    plt.plot([-0.5,1.5], [50, 50], '--k')
    
    plt.plot([0, 1], [percent_seen_real, percent_seen_catch], 
             color = [.5, .5, .5], lw = 0.5);
    plt.scatter(np.random.normal(loc = 0., scale = 0.08, size = 11), 
                percent_seen_real, s = 50, color = colors['Rec'],
                edgecolor = colors['RecDark'], zorder = 20,clip_on=False)
    plt.scatter(np.random.normal(loc = 1., scale = 0.08, size = 11), 
                percent_seen_catch, s = 50, color = [.9, .9, .9], 
                edgecolor = colors['RecDark'], zorder = 20,clip_on=False) 
    plt.xticks(range(2), ['Real', 'Catch'], rotation = 0, fontsize=13)
    plt.locator_params(axis='y', nbins=6)
    #ax.set_xticks( (-0., 1.))
    ax.set_xlim([0., 1.]);
    plt.ylabel('% Seen')
    plt.ylim([-0.01, 100]);plt.xlim([-.45, 1.45]);
 
    props = {'connectionstyle':'bar','arrowstyle':'-',\
                     'shrinkA':5,'shrinkB':5}
    mx = percent_seen_real.max() + 5
    ax.annotate('***', xy=(0.5, mx + 10), zorder=10, ha = 'center')
    ax.annotate('', xy=(0, mx), xytext=(1, mx), arrowprops=props)
    
    fig.savefig(figures_dir + '/Fig1B_1.png', 
                bbox_inches = 'tight', transparent=True)

def print_percentcorrect_bydetection_Baria():
    '''Percent correct category reports by trial type
    real/scrambled and recognized/unrecognized'''
    correct_R_real = loadmat(data_dir + 'Baria_percentcorrect_seen.mat')['p_corr_seen'][0,:]
    correct_U_real = loadmat(data_dir + 'Baria_percentcorrect_unseen.mat')['p_corr_unseen'][0,:]
    
    xdata = [0, 1]
    data = [correct_R_real, correct_U_real]
    fig, ax = plt.subplots(1, 1,figsize = (1.5, 2.5))
    ax.spines['left'].set_position(('outward', 10))
    ax.yaxis.set_ticks_position('left')
    ax.spines['bottom'].set_position(('outward', 15))
    ax.xaxis.set_ticks_position('bottom')
    box1 = plt.boxplot(data, positions = xdata, patch_artist = True, widths = 0.8,
             showfliers = False, boxprops = None, showbox = None, 
             whis = 0, showcaps = False)
    box1['boxes'][0].set( facecolor = colors['Corr'], lw=0, zorder=0)    
    box1['boxes'][1].set( facecolor = colors['Incorr'], lw=0, zorder=0)    

    box1['medians'][0].set( color = colors['CorrDark'], lw=2, zorder=20)
    box1['medians'][1].set( color = colors['IncorrDark'], lw=2, zorder=20)

    plt.plot(xdata[:2], [correct_R_real, correct_U_real],
             color = [.5,.5,.5], lw = 0.5);
      
    plt.plot([-0.5,1.5], [50, 50], '--k')

    plt.scatter(np.random.normal(loc = 0, scale = 0.08, size = correct_R_real.size), 
                correct_R_real, s = 50, color = colors['Corr'],
                edgecolor = colors['CorrDark'], zorder = 20)
    plt.scatter(np.random.normal(loc = 1, scale = 0.08, size = correct_U_real.size), 
                correct_U_real, s = 50, color = colors['Incorr'],
                edgecolor = colors['IncorrDark'], zorder = 20)
    plt.xticks(xdata, ['Y', 'N'],fontsize=15)
    plt.yticks(fontsize=15)
    #plt.xlabel('Recognition')
    plt.ylabel('% Correct',fontsize=15)
    plt.ylim([25, 110]); plt.xlim([-0.5, 1.5]);
    plt.locator_params(axis='y', nbins=5)

    props = {'connectionstyle':'bar','arrowstyle':'-',\
                     'shrinkA':4,'shrinkB':4}
    mx = correct_R_real.max() + 1
    ax.annotate('***', xy=(0.5, mx +9), zorder=10, ha = 'center')
    ax.annotate('', xy=(0, mx+1), xytext=(1, mx+1), arrowprops=props)
    plt.text(0.75,80,"***",fontsize=12)
  
    fig.savefig(figures_dir + 'Fig1B_2.png', dpi=1000, 
                transparent=True, bbox_inches = 'tight')
    