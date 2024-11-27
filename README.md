This files corresponds to the contents and organization of the 'scripts' folder.

1) 01_Preprocessing

  A) Low-Level dataset (LL)

    - ‘epochtfr_allsubs.m’, concatenates raw MEG data across blocks by trial type (seen or unseen), filtered to full band or SCP band for all subjects; writes concatenated preprocessed file for each subject

    - ‘epoching_prestim_2sec.m’, epochs data from -2 to + 3 s around stimulus onset, selects trials of interest for the analysis.

    - ‘epoching_prestim_SCP.m’ does the same as b) for the SCP band for real and catch trials.


  B) High-Level dataset (HL)

    - ‘HLTP_eye_tracker.py’, reads in eyetracking data and identifies blinks; tags trials with a blink for removal from MEG data analysis

    - ‘HLTP_main_preproc.py’, reads in raw data, runs ICA, rejects data based on ICA, runs filtering on clean data

    - ‘HLTP_make_epochs.py’, reads in cleaned and preprocessed data, rejects trials based on eyetracking data, then saves epoched data from -2 to +2 s around stimulus onset, selects trials of interest for the analysis.



2) 02_Analysis

  A) Low-Level dataset (LL)

    - ‘01_RunWavelet.m’, runs wavelet on the timecourse data and output power spectra for each subject and condition

    - ‘02_RunFooof.m’, runs spectral parametrization with fooof toolbox, output AUC alpha power, AUC beta power, as well as alpha and beta center frequencies and bandwidths for all trials

    - ‘03_RunSCPDecoding.m’, runs SVM classification based on SCP data decoding seen vs unseen trials, outputs SCP decision variable for every trial

    - ‘04a_RunClustering_alpha.m’ and ‘04b_RunClustering_beta.m’, runs cluster-based permutation analysis for alpha and beta power data; option to plot topos

    - ‘05_RunCorrelations.py’, runs correlations across trials between alpha power and SCP dv, beta power and SCP dv, alpha power and beta power; computes the AUROC scores for alpha vs beta and their residuals.

  B) High-Level dataset (HL)

    - ‘01a_RunWavelet_RunFooof.py’, extracts power spectra from wavelet, and with run_fooof(subject, use_400 = True) runs the fooof algorithm on 100 ms temporal resolution power spectra that are first averaged into the 400 ms time windows of interest in the pre-stimulus interval. Writes AUC power for each subject. The first time this script is run in its entirety, also need to run lines 164 onwards, which ensure that all subjects have the same number of sensors (272).

    - ‘01b_RunWavelet_RunFooof_scrambled.py’, same as above but for scrambled trials

    - ‘02_PrepareClusterAnalysis.py’, organizes the data into seen and unseen trials and places the data into arrays the correct size and format for cluster analysis in Matlab

    - ‘03_ClusterAnalysis.m’, runs cluster-based permutation analysis for alpha and beta power data; option to plot topos

    - ‘04_RunSCPDecoding.py’, runs logistic regression classification based on SCP data decoding recognized vs unrecognized trials, outputs SCP decision variable for every trial

    - ‘05_RunCorrelations.py’, runs correlations across trials between alpha power and SCP dv, beta power and SCP dv, alpha power and beta power; computes the AUROC scores for alpha vs beta and for beta vs SCP and their residuals. Runs mediation analysis.


3) 03_Reporting+Visualization

    - Fig1BD.py
    - Fig2AB_Fig3A_LL_timecourses.py
    - Fig2B_LL_topos.m
    - Fig2C_Fig3B_LL_scatter_box.py
    - Fig2DE_Fig3C_HL_timecourses.py
    - Fig2E_Fig3C_HL_topos.py
    - Fig2F_Fig3D_HL_scatter.py
    - Fig3A_LL_topos.py
    - Fig4BC.py
    - Fig5A.py
    - Fig6.py
    - Fig7.py


4) Supporting_files_toolboxes: a series of scripts and functions used during the pre-processing and analysis.

