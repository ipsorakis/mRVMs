%% mRVM1
% authors: Ioannis Psorakis psorakis@gmail.com, Theodoros Damoulas
% theo@dcs.gla.ac.uk
%    Copyright NCR, 2009
%
%    Use of this code is subject to the Terms and Conditions of the Research License Agreement,
%    agreed to when this code was downloaded, and a copy of which is available at
%    http://www.dcs.gla.ac.uk/inference/pMKL/Terms_and_Conditions.html.
%
% INPUTS:
% =======
% USE '-i' as input_flag if you want to provide the inputs interactively
% through the console
% OR
% use '-p' as input_flag if you want to provide the inputs as function
% parameters (see function declaration above)
%
% In case you use '-p' please provide the following:
% folds: the number of folds in the cross-validation scheme
% X_collection: cell array of length Sources, containing:
% { [N x D1] ; [N x D2] ; ...} (N samples, Di features) is the training set
% from multiple feature spaces.
% features) is the training set.
% t: of size C x N (C classes, N samples) is the training labels.
% standardize_flag: boolean cell array {Sources x 1} turns ON/OFF data
% standardization for each source
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
%
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

function OUTPUTS = cv_mRVM1_MKL(input_flag,folds,X_collection,t,standardize_flag,convergence_used,Nmax,kernel_type,kernel_param,plot_flag,dataset_name)

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
    
    Sources = input('please enter number of kernels > ');
    
    X_collection=cell(Sources,1);
    
    kernel_type = cell(Sources,1);
    kernel_param = cell(Sources,1);
    
    standardize_flag = cell(Sources,1);
    
    i=1;
    while i<=Sources
        %% LOAD DATA
        try
            fprintf(1,'\n Please provide the variable name of the %d train source',i);
            X_collection{i} = input(' > ');
            
            standardize_flag{i} = logical(input('standardize data? (1/0) > '));
            
            fprintf(1,'\n\tplease provide the kernel type of the %d source (gaussian,polynomial,linear)',i);
            kernel_type{i} = input(' > ','s');
            
          
            if ~strcmp(kernel_type{i},'linear')
                fprintf(1,'\n\tplease provide the kernel parameter of the %d source',i);
                kernel_param{i} = str2double(input(' > ','s'));
            else
                kernel_param{i} = '';
            end
            
            i=i+1;
        catch ME3
            disp('incorrect input, please enter again')
        end
    end
    
    t = input('please provide the train labels variable name > ');
    
    %
    fprintf('Please select convergence criterion:\n 1. trivial change in hyperpriors \n 2. trivial change in hyperpriors and minimum number of iterations N \n 3. Maximum iterations\n');
    convergence_used = input('choice (1 or 2 or empty if not sure) > ');
    
    if isempty(convergence_used)
        convergence_used = 2;
    elseif (convergence_used == 2) || (convergence_used == 3)
        Nmax = input('Enter maximum number of iterations (leave empty for Ntrain) > ');
    end
    %
    
    plot_flag = input('plot relevant vectors progresion? (1/0) > ');
    
end
Sources = length(X_collection);
%% CROSS VALIDATION
N = size(X_collection{1},1);
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
    
    X_train_cv = cell(Sources,1);
    for s=1:Sources
        X_train_cv{s} = X_collection{s}(XtrainIndices,:);
    end
    
    if ~exist('Nmax','var') || isempty(Nmax)
        Nmax = N1;
    end
    
    % run mRVM1
    OUTPUTS{i_run+1} = train_mRVM1_MKL('-p',X_train_cv,t(:,XtrainIndices),...
        standardize_flag,convergence_used,Nmax,kernel_type,kernel_param,plot_flag,dataset_name);
    
    disp('Number of test samples: ')
    N-N1
    
    X_test_cv = cell(Sources,1);
    for s=1:Sources
       X_test_cv{s} = X_collection{s}(XtestIndices,:); 
    end
    
    [class_membership_probabilities recognition_accuracy] = predict_mRVM_MKL(OUTPUTS{i_run+1},X_test_cv,t(:,XtestIndices))
    
    accuracies(i_run+1) = recognition_accuracy;
    RVs(i_run+1) = length(OUTPUTS{i_run+1}.active_sample_original_indices);
end
%% calculate totals

fprintf('Mean recognition accuracy: %.2f\n',mean(accuracies));
fprintf('\t +/-: %.2f\n',std(accuracies));

fprintf('Mean number of relevant vectors: %.2f\n',mean(RVs));
fprintf('\t +/-: %.2f\n',std(RVs));
end