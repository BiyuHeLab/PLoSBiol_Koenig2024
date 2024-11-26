#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 10:48:45 2017

HLTP main paradigm automatic preprocessing

@author: podvalnye
"""

import mne
import HLTP
from HLTP import subjects, MEG_pro_dir, MEG_raw_dir
from mne.preprocessing import ICA, read_ica
from HLTP_bad_ica_components import bad_comps
from matplotlib import pyplot as plt

# this line is needed to fix strange error while reading ctf files : 
import locale
locale.setlocale(locale.LC_ALL, "en_US")

def raw_block_preproc_autostage1(filename, subject):
    
    raw = mne.io.read_raw_ctf(filename, preload=True)     
    raw.info['bads'] = HLTP.bads[subject]
    raw.pick_types(meg=True, ref_meg=False, exclude='bads')    
    # TODO: remove and repair jumps if needed?
        
    # ICA
    # filter the data, only for ICA purposes
    raw.filter(l_freq=HLTP.ica_lo_freq, h_freq=HLTP.ica_hi_freq, 
               phase=HLTP.ica_phase)
        
    ica = ICA(n_components=HLTP.ica_n_com, method=HLTP.ica_method,
              random_state=HLTP.ica_random_state)
    
    ica.fit(raw, decim=HLTP.ica_decim, reject=dict(mag=4.5e-12))
    
    block_name = HLTP.get_block_type(filename)
    fdir = MEG_pro_dir + '/' + subject + '/'+ block_name    
    ica.save(fdir + '-ica.fif')
        
    # plot_ica
    fdir = MEG_pro_dir + '/' + subject + '/figures/'+ block_name
    plot_ica_timecourse(ica, raw, subject, fdir)
    plot_ica_components(ica, subject, fdir)
  
def plot_ica_timecourse(ica, raw, subject, fdir):
    ica_range=range(0,49,10)
    n_courses2plot=10
    ica_sources=ica.get_sources(raw)
    source_data= ica_sources.get_data()
    for i in ica_range:
        plt.figure(figsize=(20,20))
        plt.title(subject + "Components %s through %s" %(i,i+n_courses2plot))
        for courses in range(i , i+n_courses2plot):
            plt.plot(10*courses+source_data[courses][:], linewidth=0.2)
            plt.text(-15000, courses*10, '%s' %(courses))
        plt.savefig(fdir + "_timecourses_%s.png" %(courses),dpi=300)        
        plt.close()
        
def plot_ica_components(ica, subject, fdir):
    ica_range=range(0,49,10)
    n_components2plot=10
    for i in ica_range:
        picks= range(i,i+n_components2plot)
        comps=ica.plot_components(picks)
        picks_min= str(picks[0])
        picks_max= str(picks[-1])
        comps.savefig(fdir + '_ica_components_' + picks_min + '_through_'
                    +picks_max+ '.png') 
        plt.close(comps)
    
# apply ica rejection and save the clean data
def raw_block_preproc_autostage2(filename, subject):
    block_name = HLTP.get_block_type(filename)
    fdir = MEG_pro_dir + '/' + subject + '/'+ block_name
    raw = mne.io.read_raw_ctf(filename, preload=True)       

    ica = read_ica(fdir + '-ica.fif')
    ica.exclude = bad_comps[subject][block_name]   
    ica.apply(raw)
    
    raw.save(fdir + '_stage2_raw.fif', overwrite=True)   
    
# run automatic stage 1 for all subjects   
for s in subjects:      
    subj_raw_dir = MEG_raw_dir + '/' + s + '/'
    filenames, subj_code, n_blocks, date = \
        HLTP.get_experimental_details(subj_raw_dir)    
    for f in filenames:
        raw_block_preproc_autostage1(f, s)

# run automatic stage 2 for all subjects, after Explore the ICA figures    
for s in subjects:
    subj_raw_dir = MEG_raw_dir + '/' + s + '/'
    filenames, subj_code, n_blocks, date = \
        HLTP.get_experimental_details(subj_raw_dir) 
    for f in filenames:
        if f.find('thresh') > 0:
            raw_block_preproc_autostage2(f, s)  

