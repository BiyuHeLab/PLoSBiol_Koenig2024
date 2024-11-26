function [svm_acc, models, activation_patterns, DVs] = libsvm_nfold_lk(group, data, n_folds, n_reps, libsvm_options_train, libsvm_options_test, scale_data)
% [svm_acc, models, activation_patterns] = libsvm_nfold(group, data, n_folds, n_reps, libsvm_options_train, libsvm_options_test, scale_data)
% 
% Decode the labels in "group" using the features in "data" with the libsvm
% toolbox.
%
% INPUTS
% ------
% group   - label for each trials
% data    - n_trials x n_attributes matrix
% n_folds - number of training/testing folds
% n_reps  - number of repetitions of balanced training subsets in each fold
% libsvm_options_train - libsvm option string for SVM training
% libsvm_options_test  - libsvm option string for SVM testing
% scale_data - if 1, data in each feature is rescaled to [-1, 1] for each
%              testing set. [default = 0]
% 
% OUTPUTS
% -------
% svm_acc - svm accuracy for each fold x repetition
% models  - fold x repetition cell array holding the SVM models as output
%           by libsvmtrain for each fold and repetition 
% activation_patterns - fold x repetition cell array holding the activation 
%           pattern for each fold and repetition 
% DVs - average distance from the decision hyperplane for each trial in
%           each fold across repetitions
% ind_fold - for each fold, the indice in the original dataset of the
%           trials included. corresponds to DVs



if ~exist('scale_data', 'var') || isempty(scale_data)
    scale_data = 0;
end


n_trials = length(group);

% define fold indeces by interleaving consecutive trials
for i_fold = 1:n_folds
    ind_fold{i_fold} = i_fold : n_folds : n_trials;
end


% perform training and testing for each fold
for i_fold = 1:n_folds
    
    ind_test  = ind_fold{i_fold};
    ind_train = setdiff(1:n_trials, ind_test);

    % define training and testing sets
    data_train = data(ind_train, :);
    data_test  = data(ind_test, :);

    group_train = group(ind_train);
    group_test  = group(ind_test);


    % count number of trials for each group level to allow for svm
    % training that has balanced number of trials for each group level
    ng1 = sum(group_train ==  1);
    ng2 = sum(group_train == -1);

    if ng1 < ng2, ngmin = ng1; else ngmin = ng2; end


    % repeatedly take a random, balanced subset of training trials,
    % train the classifier, and test classification performance on the
    % test set
    for rand_rep = 1:n_reps

        % get random balanced set for training data group 1
        data_train1  = data_train(group_train == 1, :);
        group_train1 = group_train(group_train == 1);

        ind_rand1 = randperm(length(group_train1));
        ind_bal1  = ind_rand1(1:ngmin);
        data_train1  = data_train1(ind_bal1, :);
        group_train1 = group_train1(ind_bal1);

        % get random balanced set for training data group 2
        data_train2  = data_train(group_train == -1, :);
        group_train2 = group_train(group_train == -1);

        ind_rand2 = randperm(length(group_train2));
        ind_bal2  = ind_rand2(1:ngmin);
        data_train2  = data_train2(ind_bal2, :);
        group_train2 = group_train2(ind_bal2);

        % concatenate
        data_train_i = [data_train1; data_train2];
        group_train_i = [group_train1; group_train2];

        % train and classify
        model = libsvmtrain(group_train_i, data_train_i, libsvm_options_train, scale_data);
        [predicted_group, accuracy, decision_values]  = libsvmpredict_lk(group_test, data_test, model, libsvm_options_test);        
        
        svm_acc(i_fold, rand_rep) = mean(predicted_group == group_test);
        models{i_fold, rand_rep} = model;
        DVs_all{i_fold, rand_rep} = decision_values;
        
        % calculate activation pattern
        W = model.sv_coef' * model.SVs;
        train_cov = cov(data_train_i); 
        activation_patterns{i_fold, rand_rep} = train_cov * W';
        
    end
    DVs_fold{i_fold} = mean(cell2mat(DVs_all(i_fold,:)),2);

end
DVs = cell2mat(DVs_fold(:));
[indices, sortorder] = sort(cell2mat(ind_fold));
DVs = DVs(sortorder);
end

% distrib_DVs = zeros(22,10);
% for j = 1:10
%     distrib_DVs(:,j) = DVs_all{1,j};
% end
% 
% %Plot histogram for indiividual rows (= individual trials)
% histogram(distrib_DVs(1,:),10)
% histogram(distrib_DVs(2,:),10)
% histogram(distrib_DVs(3,:),10)
% histogram(distrib_DVs(5,:),10)
% histogram(distrib_DVs(6,:),10)
% histogram(distrib_DVs(12,:),10)
% histogram(distrib_DVs(21,:),10)
% histogram(distrib_DVs(16,:),10)
% histogram(distrib_DVs(14,:),10)


