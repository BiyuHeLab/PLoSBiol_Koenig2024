#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 12:25:31 2017

Prepare epochs structures using clean data, do both raw and scp

@author: podvalnye
"""
import mne
import HLTP
from scipy import signal
import numpy

def get_detrended_raw(fname):
    raw = mne.io.read_raw_fif(fname, preload=True)
    raw.info['bads'] = HLTP.bads[subject]    
    picks = mne.pick_types(raw.info, meg = True, 
                           ref_meg = False, exclude = 'bads')
    raw.apply_function(signal.detrend, picks=picks, dtype=None, n_jobs=24)
    return raw, picks

def mov_average(x):
    window_len = int(1200 * 0.2)
    w = numpy.ones(window_len,'d')    
    y = numpy.convolve(w/w.sum(), x, mode = 'same')
    return y

def localizer_raw_to_epochs(subject):
    fdir = HLTP.MEG_pro_dir + '/' + subject
    
    raw, picks = get_detrended_raw(fdir + '/localizer_stage2_raw.fif')
    
    raw.filter(l_freq=None, h_freq=35.0, picks=picks, 
               filter_length=HLTP.filter_length, phase = HLTP.filter_phase, 
               method = HLTP.filter_method, h_trans_bandwidth = 10) 
    events = mne.find_events(raw, HLTP.stim_channel, 
                             mask = HLTP.event_id['Stimulus'], mask_type = 'and')
    events[:, 2] = HLTP.get_subj_loc_events(subject)
    event_ids = HLTP.category_id

    picks = mne.pick_types(raw.info, meg=True, exclude='bads')
    epochs = mne.Epochs(raw, events, event_ids, tmin=-0.1, tmax=0.6,  
                            reject=dict(mag=1e-11), baseline=(-0.1, 0), 
                            picks=picks, preload=True, verbose=False, detrend=0)
    epochs.save(fdir + '/localizer_lpf_35Hz-epo.fif')


def main_raw_to_epochs(subject, freq = None, func = None, file_name = ''):
    '''
    Apply any lowpass filter or any other function on the data and epoch it. 
    '''
    tmin, tmax = -2, 2
    block_epochs = [];
    subj_raw_dir = HLTP.MEG_raw_dir + '/' + subject + '/'
    fdir = HLTP.MEG_pro_dir + '/' + subject
    _, _, n_blocks, _ = HLTP.get_experimental_details(subj_raw_dir) 
    decim = 1 # do not decimate data by default
    for b in range(1, n_blocks + 1):
        raw, picks = get_detrended_raw(fdir + '/thresh' + "%02d" % b + 
                                       '_stage2_raw.fif')
        events = mne.find_events(raw, stim_channel=HLTP.stim_channel, 
                                      mask = HLTP.event_id['Stimulus'], 
                                      mask_type = 'and')
        # Correct for photo-diode delay:
        events[:, 0] = events[:, 0] + HLTP.PD_delay
        if func:
            raw.apply_function(func, picks=picks, dtype=None, n_jobs=24)
            # note: never save picks with epochs because they will not be readable                    
        if freq:
            raw.filter(l_freq=None, h_freq=freq, picks=picks, 
                   filter_length=HLTP.filter_length, phase = HLTP.filter_phase, 
                   method = HLTP.filter_method, h_trans_bandwidth = 10, n_jobs=24)
            decim = 12            
        epochs = mne.Epochs(raw, events, {'Stimulus': 1}, tmin=tmin, tmax=tmax, 
                        proj=True, baseline=None, preload=True, detrend=0, 
                        verbose=False, decim = decim)
        block_epochs.append(epochs)    
    #Concatenate    
    #lie about head position, otherwise concatenation doesn't work:
    for b in range(0, n_blocks):
        block_epochs[b].info['dev_head_t'] = block_epochs[0].info['dev_head_t']
    
    all_epochs = mne.concatenate_epochs(block_epochs)
    
    # remove trials when the eyes were closed on stimulus presentation
    bad_trials = HLTP.load(HLTP.MEG_pro_dir + '/' + subject + '/bad_trials.p')
    if len(bad_trials) < 40:
        all_epochs.drop(bad_trials)
    #Do not interpolate, otherwise mne can't read missing channels   
    all_epochs.save(HLTP.MEG_pro_dir + '/' + subject + '/' + file_name + '.fif') 

#for subject in HLTP.subjects:
#    localizer_raw_to_epochs(subject)       
# epoch the data and concatenate the blocks 
for subject in HLTP.subjects:     
    #main_raw_to_epochs(subject, freq = 35.0, file_name='HLTP_0_35_Hz_stim-epo')
    #main_raw_to_epochs(subject, freq = 5.0, file_name='HLTP_0_5_Hz_stim-epo')
    #main_raw_to_epochs(subject, freq = None, func = mov_average, 
    #                   file_name = 'HLTP_movavg200ms_stim-epo')
    main_raw_to_epochs(subject, freq = None, func = None, 
                       file_name='HLTP_raw_stim-epo')
    