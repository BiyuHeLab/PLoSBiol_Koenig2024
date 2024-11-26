#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 10:58:50 2018

Decode subjective recognition using pre-stimulus MEG signal

@author: podvae01
"""
import os
os.chdir('/isilon/LFMI/VMdrive/Lua/MEG_alpha_1-f_SCP_PLoSBiol_2024/scripts')

import HLTP
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, \
    recall_score
import numpy as np
import pandas as pd
from scipy.special import logit

data_dir = '/isilon/LFMI/VMdrive/Lua/MEG_alpha_1-f_SCP_PLoSBiol_2024/data/HL/'
figures_dir = '/isilon/LFMI/VMdrive/Lua/MEG_alpha_1-f_SCP_PLoSBiol_2024/figures/'
supp_dir = '/isilon/LFMI/VMdrive/Lua/MEG_alpha_1-f_SCP_PLoSBiol_2024/scripts/supporting_files_toolboxes/HL/'

times = np.arange(-2.0,1.6,0.1)
epoch_name = 'HLTP_raw_stim'
all_data = []
all_trial_n = []
for subject in HLTP.subjects:
    epochs = HLTP.get_raw_epochs(subject, epoch_name) 
    epochs.crop(tmin = -2.0, tmax = 1.5)
    data = np.zeros((epochs.get_data().shape[0], epochs.get_data().shape[1], len(times)))
    for i, t_start in enumerate(times):
        t_end = t_start + 0.1
        indices = np.where((epochs.times >= t_start) & (epochs.times < t_end))[0]
        data_t = epochs.get_data()[:, :, indices].mean(axis=-1)
        data[:,:,i] = data_t
    trial_n = epochs.selection
    all_data.append(data)
    all_trial_n.append(trial_n)

HLTP.save([all_data, all_trial_n], data_dir + 'pre+poststim_data_for_MVPA.p')

# model definition
model = make_pipeline(StandardScaler(), LogisticRegression(C = 1))
cv = LeaveOneOut()

# We define two broad ROIs, occipitotemporal and frontoparietal
sensors = {}
for s in ['O','F','T','P','C']:
    sensors[s] = HLTP.get_ROI('', s)[1]

sensors['OT'] = np.hstack([HLTP.get_ROI('', 'O')[1], HLTP.get_ROI('', 'T')[1]])
sensors['FP'] = np.hstack([HLTP.get_ROI('', 'F')[1], HLTP.get_ROI('', 'P')[1]])
sensors['AL'] = HLTP.get_ROI('', 'A')[1]

# Load data that contains 2 s pre-stim data mean, all subjects, the data has
# certain trials removed due to meg artifacts and eye blinks. 
all_data, trial_n = HLTP.load(data_dir + 'pre+poststim_data_for_MVPA.p')

K_PERM = 1000 # number of permutations 
N_JOBS = -1   # number of parallel jobs whenever possible 

#---------------------- Prepare data frame for results ------------------------
# we will record all findings in this data structure
scores = ['train_img', 'train_cat', 'test_img', 'test_cat', 'cv', 'auroc', 
          'subject', 'decoded_variable', 'decoder', 'roi']
scores_dict = dict((key, []) for key in scores)
scores_dict['cv'] = "%s" % cv
scores_dict['decoded_variable'] = ['recognition']
scores_dict['decoder'] = ['Logisticegression']
scores_list = []

def record_scores(scores_dict, y_true, y_prob, sub, roi, 
                  test_img_type, train_img_type, test_cat_name, train_cat_name,
                  perm_auroc):
    
    scores_dict['subject'] = [sub]
    scores_dict['roi'] = [roi]
    scores_dict['train_img'] = [train_img_type]
    scores_dict['train_cat'] = [train_cat_name]  
    scores_dict['test_img'] = [test_img_type]
    scores_dict['test_cat'] = [test_cat_name]
 
    y_pred = -1. * (y_prob[:, 0] > 0.5) + 1. * (y_prob[:, 1] > 0.5)
    scores_dict['auroc'] = roc_auc_score(y_true, y_prob[:, 1])                       
    scores_dict['f1_weighted'] = f1_score(y_true, y_pred, 'weighted')
    scores_dict['f1_macro'] = f1_score(y_true, y_pred, average = 'macro')
    scores_dict['balanced_acc'] = recall_score(
            y_true, y_pred, pos_label=None,average='macro')
    scores_dict['accuracy'] = accuracy_score(y_true, y_pred) 
    for idx, key in enumerate(perm_auroc): 
        scores_dict['perm_score' + str(idx)] = key
    return scores_dict

def CVpermutest(K, func, model, data, labels):
    perm_auroc = []
    for k in range(K):
        perm_pred_prob = func(model, data, np.random.permutation(labels), 
                        cv = cv, method = 'predict_proba', n_jobs = N_JOBS)
        perm_auroc.append(roc_auc_score(labels, perm_pred_prob[:, 1]))
    return perm_auroc

def CCpermutest(K, func, model, data_train, 
                labels_train, data_test, labels_test):
    perm_auroc = []
    for k in range(K):
        perm_pred_prob = fitpredict(model, data_train, 
                                    np.random.permutation(labels_train), 
                                    data_test)
        perm_auroc.append(roc_auc_score(labels_test, perm_pred_prob[:, 1])) 
    return perm_auroc
  
def fitpredict(model, data_train, labels_train, data_test):
    model.fit(data_train, labels_train)
    return model.predict_proba(data_test)
                
remove_trials_s16 = np.array([41,72,113,137,149,156,160,167,195])

#----------------- Decode recognition within/cross real/scrambled -------------
#Use all trials of real and scrambled
data = {'real':[], 'scr':[], 'all':[]}; labels = {'real':[], 'scr':[], 'all':[]};
trialIDs = {'real':[], 'scr':[], 'all':[]}

data['real'], labels['real'], trialIDs['real'] = HLTP.prep_data_and_label(all_data, trial_n, 
                'recognition', {'real_img':True}, label_values = [1, -1])
data['real'][16] = np.delete(data['real'][16],remove_trials_s16,axis=0)
labels['real'][16] = np.delete(labels['real'][16],remove_trials_s16)

data['scr'], labels['scr'], trialIDs['scr'] = HLTP.prep_data_and_label(all_data, trial_n, 
                'recognition', {'real_img':False}, label_values = [1, -1])

for sub_idx in range(len(HLTP.subjects)):
    data['all'].append(np.concatenate((data['real'][sub_idx],data['scr'][sub_idx])))
    labels['all'].append(np.concatenate((labels['real'][sub_idx],labels['scr'][sub_idx])))
    trialIDs['all'].append(np.concatenate((trialIDs['real'][sub_idx],trialIDs['scr'][sub_idx])))

permu_auroc = []
roi = 'wb' #whole brain
test_img = 'all'
train_img = 'all'
pred_prob_all = []
pred_prob_all_logit = []
n_reps = 10

for sub_idx, sub in enumerate(HLTP.subjects):
    pred_prob_allt = np.zeros((data[train_img][sub_idx].shape[0],2,data[train_img][sub_idx].shape[2],n_reps)) #trials x (2 columns for 2 classes) x time x repetitions
    pred_prob_allt[:] = np.nan
    for time in range(35):
        n_group1 = sum(labels[train_img][sub_idx] == 1)
        n_group2 = sum(labels[train_img][sub_idx] == -1)
        
        # Check we have enough trials
        if n_group1 < 5 or n_group2 < 5:
                continue;
                
        #Balance the training set
        if n_group1 < n_group2:
            n_min = n_group1
        else:
            n_min = n_group2
        
        indices_labels_g1 = np.where(labels[train_img][sub_idx] == 1)[0]
        labels_g1 = labels[train_img][sub_idx][indices_labels_g1]
        data_g1 = data[train_img][sub_idx][indices_labels_g1,:,time]
        indices_labels_g2 = np.where(labels[train_img][sub_idx] == -1)[0]
        labels_g2 = labels[train_img][sub_idx][indices_labels_g2]
        data_g2 = data[train_img][sub_idx][indices_labels_g2,:,time]
        
        #Repeatedly take a random, balanced subset of training trials, train the classifier, and test classification performance on the test set
        for rep in range(n_reps):
            #Get random balanced sets for both classes
            indices_rand1 = np.random.permutation(len(indices_labels_g1))
            indices_balanced1 = indices_rand1[:n_min]
            data_bal_g1 = data_g1[indices_balanced1,:]
            labels_bal_g1 = labels_g1[indices_balanced1]
            indices_bal_g1 = indices_labels_g1[indices_balanced1] #Indices in full dataset pre-separation into classes and pre-balancing
            
            indices_rand2 = np.random.permutation(len(indices_labels_g2))
            indices_balanced2 = indices_rand2[:n_min]
            data_bal_g2 = data_g2[indices_balanced2,:]
            labels_bal_g2 = labels_g2[indices_balanced2]
            indices_bal_g2 = indices_labels_g2[indices_balanced2] #Indices in full dataset pre-separation into classes and pre-balancing            
            
            #Concatenate
            data_train_balanced = np.concatenate((data_bal_g1, data_bal_g2),axis=0)
            labels_train_balanced = np.concatenate((labels_bal_g1,labels_bal_g2))
            final_indices_balanced = np.concatenate((indices_bal_g1,indices_bal_g2))

            # Cross-validate
            pred_prob = cross_val_predict(model, data_train_balanced, labels_train_balanced, cv = cv, method='predict_proba', n_jobs = N_JOBS)
            
            pred_prob_allt[final_indices_balanced,:,time,rep] = pred_prob
            
       
    pred_prob_allt_bal = np.nanmean(pred_prob_allt,axis=3)    
    pred_prob_allt_logit = logit(pred_prob_allt_bal)
    pred_prob_all.append(pred_prob_allt_bal)
    pred_prob_all_logit.append(pred_prob_allt_logit)
                                
import pickle
with open(data_dir + 'DecisionVariables.pkl', 'wb') as f:
    pickle.dump(pred_prob_all, f)

with open(data_dir + 'DecisionVariables_logit.pkl', 'wb') as f:
    pickle.dump(pred_prob_all_logit, f)
    
#The trial IDs associated with both of the above are saved in trialIDs['all']
with open(data_dir + 'trial_indices_decisionvariables.pkl', 'wb') as f:
    pickle.dump(trialIDs['all'], f)