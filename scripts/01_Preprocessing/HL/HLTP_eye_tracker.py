#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 10:31:00 2017

@author: podvalnye
"""
import mne
import HLTP
import numpy as np

def get_blinks(pupil_data):
    thr = -3.6;             
    pupil_data[0] = thr;
    blinks = np.diff((pupil_data < thr).astype(int), n = 1, axis = 0)
    blink_start = (blinks == 1).nonzero()[0]
    blink_end = (blinks == -1).nonzero()[0]
    
    if len(blink_start) > len(blink_end):
        blink_end=np.append(blink_end, len(pupil_data))
    return blink_start, blink_end

# Find all trials with a blink interferance of stimulus presentation
def get_bad_trials(raw, blink_start, blink_end):
    stim_duration = 0.1 * raw.info['sfreq']
    events = mne.find_events(raw, stim_channel=HLTP.stim_channel,
                             mask = HLTP.event_id['Stimulus'], 
                             mask_type = 'and')[:, 0]
    blink_duration = blink_end - blink_start
    short_blinks =  np.where(blink_duration < 0.05 * raw.info['sfreq'])[0]
    long_blinks =  np.where(blink_duration > 4 * raw.info['sfreq'])[0]
    blinks_to_remove = np.concatenate([short_blinks,long_blinks])
    blink_end = np.delete(blink_end, blinks_to_remove)
    blink_start = np.delete(blink_start, blinks_to_remove)
    bad_trials = []
    for idx, evnt in enumerate(events):
        blink_starts_before = (blink_start - evnt) < stim_duration # blink starts before stimulus offset
        blink_ends_after = (blink_end - evnt) > 0 # blink ends after stimulu onset
        if any(np.logical_and(blink_starts_before, blink_ends_after)):
            bad_trials.append(idx)
    return bad_trials

# extract raw pupil data and save
for s in HLTP.subjects:
    subj_raw_dir = HLTP.MEG_raw_dir + '/' + s + '/'
    subj_pro_dir = HLTP.MEG_pro_dir + '/' + s + '/'
    filenames, _, _, _ = HLTP.get_experimental_details(subj_raw_dir)    
    all_bad_trials = []
    main_bad_trials = []
    for f in filenames[0:1]:
        raw = mne.io.read_raw_ctf(f, preload=True)    
        block_name = HLTP.get_block_type(f)
        pupil_data = raw.get_data(picks=[HLTP.pupil_chan])[0]
        #HLTP.save(pupil_data, subj_pro_dir +'/' + block_name + 'raw_pupil.p')      

        if block_name.find('thr') > -1: 
            blink_start, blink_end = get_blinks(pupil_data)
            bad_trials = get_bad_trials(raw, blink_start, blink_end)
            if bad_trials:
                all_bad_trials.append(np.asarray(bad_trials) + 
                                      (int(block_name[-2:]) - 1) * 36 )
    if len(all_bad_trials) > 0:
        main_bad_trials = np.concatenate(all_bad_trials)
    HLTP.save(main_bad_trials, HLTP.MEG_pro_dir + '/' + s + '/bad_trials.p')

n_bad_trials=[]
for s in HLTP.subjects:
    fname = HLTP.MEG_pro_dir + '/' + s + '/bad_trials.p'
    n_bad_trials.append(len(HLTP.load(fname)))

n_bad_trials = np.array(n_bad_trials)
n_bad_trials = n_bad_trials[n_bad_trials < 40]
n_bad_trials.mean()
n_bad_trials.std() / np.sqrt(len(n_bad_trials))

