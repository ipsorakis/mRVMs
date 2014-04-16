%% mRVM1
% authors: Ioannis Psorakis psorakis@gmail.com, Theodoros Damoulas theo@dcs.gla.ac.uk
%    Copyright NCR, 2009
%
%    Use of this code is subject to the Terms and Conditions of the Research License Agreement,
%    agreed to when this code was downloaded, and a copy of which is available at
%    http://www.dcs.gla.ac.uk/inference/pMKL/Terms_and_Conditions.html.
% INPUTS:
% =======
% USE '-i' as input_flag if you want to provide the inputs interactively
% through the console
% OR
% use '-p' as input_flag if you want to provide the inputs as function
% parameters (see function declaration above)
% In case you use '-p' please provide the following:
% folds: the number of folds in the cross-validation scheme
% X: of size N x D (N samples, D features) is the training set.
% t: of size C x N (C classes, N samples) is the training labels.
% standardize_flag: [boolean] turns data standardization ON/OFF
% convergence_used: values [1 or 2] is the training termination criterion (see
%                   conv.1 and conv.2 of theoretical background
% kernel_type: string can be either 'gaussian', or 'polynomial' or 'linear;
% kernel_param: for linear kernel put any value
% plot_flag: [1 or 0] plots the number of relevant vectors during training
% dataset_name: auxiliary label
% In case you use '-i' please note:
% each dataset file must contain the necessary variables, which much be in the correct format. Those are:
% 
% 1 the class labels, say "t". This variable must be a C X N dimensional array
%   where C is the number of classes and N is the size of the data. Say we have a
%   problem with 100 samples and 3 classes. The 5th sample would belong to the 2nd
%   class if t(:,5)' = [0 1 0].
%   For datasets which do have independent training and test sets, there should be two of these
%   variables. E.g. tTrain and tTest
% 
% 2 the data, say "X". This variable should be a N X D where N is the number of samples
%   and D is the number of features. For datasets which do have independent training
%   and test sets, there should be two of there variables. E.g. Xtrain Xtest. Also, for multi-kernel problems
%   there should be one such variable for each feature space.
% OUTPUTS:
% =======
% OUTPUTS is a cell array containing "mRVM_train_output" objects (one per fold) that have a range of properties:
% model_used: the name of the algorithm (e.g mRVM1);
% dataset_name: the name of the dataset;
% N_total: the total number of samples in the original dataset;
% N_prototypical: the number of relevance vectors extracted from the algorithm;
% 
% X_prototypical: the relevance vectors (or prototypical samples) extracted from the algorithm;
% 
% X_prototypical_standardized: same as above, but standardized;
% K_sparse: the sparse training kernel, that uses only the relevance vectors;
% W_sparse: the sparse regressors matrix, that uses only the relevance vectors;
% 
% active_sample_original_indices: the original indices of the relevance vectors in the dataset;
% 
% sources: number of kernels;
% b: kernel mixing coefficients;
% 
% kernel_type: kernel(s) used;
% kernel_param: kernel parameter(s) used;

function OUTPUTS = cv_mRVM1(input_flag,folds,X,t,standardize_flag,convergence_used,Nmax,kernel_type,kernel_param,plot_flag,dataset_name)

disp('---------------- mRVM1 ----------------')
model_used = 'mRVM1';

if ~exist('input_flag','var')
    input_flag = '-i';
end

if strcmp(input_flag,'-i')
    %% INTERFACE
    % provide the dataset file name
    dataset_name = input('please enter the file name of the dataset > ','s');
    
    % attempt to open the file name, if not, stop
    try
        load(dataset_name);
    catch ME
        ME.stack;
        error('No dataset exists with that name, exiting...')
    end
    
    folds = input('please enter number of folds > ');
    
    X = input('please provide the training set variable name > ');
    
    standardize_flag = logical(input('standardize data? (1/0) > '));
    % KERNEL SETUP
    
    fprintf(1,'Please enter the kernel number:\n1. Gaussian\n2. Polynomial\n3. Linear\n');
    kernel_ID = input('your choice > ');
    switch kernel_ID
        case 1
            kernel_type='gaussian';
            kernel_param = input('please enter the kernel parameter > ');
        case 2
            kernel_type = 'polynomial';
            kernel_param = input('please enter the kernel parameter > ');
        case 3
            kernel_type = 'linear';
            kernel_param='';
    end
    
    t = input('please provide the train labels variable name > ');
    
    %%%
    fprintf('Please select convergence criterion:\n 1. trivial change in hyperpriors \n 2. trivial change in hyperpriors and minimum number of iterations N \n 3. Maximum iterations\n');
    convergence_used = input('choice (1 or 2 or empty if not sure) > ');
    
    if isempty(convergence_used)
        convergence_used = 2;
    elseif (convergence_used == 2) || (convergence_used == 3)
        Nmax = input('Enter maximum number of iterations (leave empty for Ntrain) > ');
    end
    
    %
    plot_flag = input('plot relevant vectors progresion? (1/0) > ');
    

%elseif strcmp(input_flag,'-f')    
%elseif strcmp(input_flag,'-p')    
end
%% CROSS VALIDATION
N = size(X,1);
Nsplit = floor(N/folds);

OUTPUTS = cell(folds,1);
accuracies = zeros(folds,1);
RVs = zeros(folds,1);

% do a random permutation of the dataset indices, and then partition the
% dataset into folds.
Xindices = randperm(N)';
for i_run=0:folds-1
    % for each fold, estimate the training and test indices, and perform
    % one train and test
    %                       
    disp(strcat('========== fold:',num2str(i_run+1),...
        '/',num2str(folds),' =========='))
    XtestIndices = Xindices(i_run*Nsplit+1:1:Nsplit*(i_run+1));
    XtrainIndices = Xindices([[1:1:i_run*Nsplit] [Nsplit*(i_run+1)+1:1:N]]);
    
    disp('Number of train samples: ')
    N1 = length(XtrainIndices)
    
    if ~exist('Nmax','var') || isempty(Nmax)
        Nmax = N1;
    end
    
    % run mRVM1
    OUTPUTS{i_run+1} = train_mRVM1('-p',X(XtrainIndices,:),t(:,XtrainIndices),...
        standardize_flag,convergence_used,Nmax,kernel_type,kernel_param,plot_flag,dataset_name);
    
    disp('Number of test samples: ')
    N-N1
    
    [class_membership_probabilities recognition_accuracy] = predict_mRVM(OUTPUTS{i_run+1},X(XtestIndices,:),t(:,XtestIndices))
    
    accuracies(i_run+1) = recognition_accuracy;
    RVs(i_run+1) = length(OUTPUTS{i_run+1}.active_sample_original_indices);
end
%% calculate totals

fprintf('Mean recognition accuracy: %.2f\n',mean(accuracies));
fprintf('\t +/-: %.2f\n',std(accuracies));

fprintf('Mean number of relevant vectors: %.2f\n',mean(RVs));
fprintf('\t +/-: %.2f\n',std(RVs));
end