%% mRVM2
% authors: Ioannis Psorakis psorakis@gmail.com, Theodoros Damoulas
% theo@dcs.gla.ac.uk
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
% X: of size N x D (N samples, D features) is the training set.
% t: of size C x N (C classes, N samples) is the training labels.
% standardize_flag: [boolean] turns data standardization ON/OFF
% convergence_used: values [1 or 2] is the training termination criterion (see
%                   conv.A and conv.N of theoretical background
% Nmax: maximum # of iterations (use size(X,1) if unsure)
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
% OUTPUT is an object that has a range of properties:
% model_used: the name of the algorithm (e.g mRVM2);
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

function OUTPUT = train_mRVM2(input_flag,X,t,standardize_flag,convergence_used,Nmax,kernel_type,kernel_param,plot_flag,dataset_name)

disp('---------------- mRVM2 ----------------')
model_used = 'mRVM2';

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
    
    %
    fprintf('Please select convergence criterion:\n 1. trivial change in hyperpriors \n 2. maximum number of iterations N \n');
    convergence_used = input('choice (1 or 2 or empty if not sure) > ');
    if isempty(convergence_used)
        convergence_used = 1;
    elseif convergence_used==2
        Nmax = input('enter maximum number of iterations (or leave empty for Ntrain) > ');
        if isempty(Nmax)
            Nmax = size(X,1);
        end
    end
    %
    
    plot_flag = input('plot relevant vectors progresion? (1/0) > ');
    
    
    %elseif strcmp(input_flag,'-f')
    %elseif strcmp(input_flag,'-p')
end

%% INITIALIZATIONS
if standardize_flag
    [Xtrain,Xmean,Xstd] = standardize_data(X);
else
    Xtrain = X;
    Xmean = 0;
    Xstd = 0;
end

N = size(Xtrain,1);
D = size(Xtrain,2);
C = size(t,1);

% initialize the auxiliary variables Y to follow the target labels of the
% training set
Y = 10*rand(C,N).*t + rand(C,N);
% set all scales to infinity
A = ones(N,C);
% set all regressor values to zero
W = randn(N,C);

if plot_flag
    number_of_RVs = zeros(N+1,1);
end

prune_flag=1;
astep='mean';
t_gamma=1e-05;
u_gamma=1e-05;
sparse_flag='sparse';

converged_loga = false;
%% PERFORM KERNEL SETUP
disp('building kernel(s)...')
Ktrain = build_standarized_kernel...
    (Xtrain,Xtrain,kernel_type,kernel_param);

% mRVM2 starts with full model
active_samples = 1:N;

mRVM2_iterator=0;
disp('training...')
tic
while true
    % store the values of the previous iteration
    number_of_active_samples_old = length(active_samples);
    A_active_old = A(active_samples,:);
    %% update A
    %update the scales of all samples which are inside the model
    A(active_samples,:) = ...
        postA(W(active_samples,:),t_gamma,u_gamma,astep);
    %% Pruning out
    % remove samples which are above a certain threshold
    active_samples = find(~prod((A>10^4)*1,2));
    % if all samples are high, keep one with the smalles A
    if isempty(active_samples)
        active_samples = find(mean(A,2) == min(mean(A,2)),C);
    end
    %% update W,Y
    % update posteriors for all samples in the model
    W(active_samples,:) = ...
        postW_map(W(active_samples,:),Ktrain(active_samples,:),Y,A(active_samples,:));
    
    % update Y
    F=Ktrain(active_samples,:)'*W(active_samples,:);
    for c=1:C
        pos=find(t(c,:)==1);
        [Z, Yt(pos,:)] = YTruncate(F(pos',:),c,1000);
    end
    Y=Yt';
    
    %% CALCULATE LOGA CONVERGENCE
    if length(active_samples)==...
            number_of_active_samples_old && ~converged_loga
        loga = abs(log(A(active_samples,:)) - log(A_active_old));
        converged_loga = prod(prod(1*(loga<1e-2)));
    end
    %% PLOTS
    if plot_flag
        number_of_RVs(mRVM2_iterator+1) = length(active_samples);
        plot(number_of_RVs(1:mRVM2_iterator+1));
        title('Relevant vectors')
        drawnow;
    end
    
    %% stop when the maximum number of iterations have been reached.
    if (convergence_used==1 && converged_loga) || (convergence_used==2 && mRVM2_iterator==Nmax)
        break;
    end
    
    % break if not converged after 2*N
    if mRVM2_iterator == 2 * N
        break;
        warning('not converged - training terminated after default maximum iterations')
    end    
    %% increment index
    mRVM2_iterator =mRVM2_iterator+1;
end

disp('TRAINING FINALIZED.');
disp('-------------------');
toc

fprintf('Total iterations until convergence: %d\n',mRVM2_iterator)
fprintf('Number of prototypical samples: %d out of %d total.\n',length(active_samples),N)

if ~exist('dataset_name','var')
    dataset_name = 'untitled_run';
end

% 
if standardize_flag
    Xtrain = Xtrain(active_samples,:);
else
    Xtrain = nan;
end

OUTPUT = mRVM_train_output(model_used, dataset_name, N, X(active_samples,:),standardize_flag,Xtrain,Xmean,Xstd, Ktrain(active_samples,active_samples),...
    W(active_samples,:), active_samples,kernel_type,kernel_param);
end
