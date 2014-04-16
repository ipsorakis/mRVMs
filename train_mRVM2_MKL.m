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
%
% In case you use '-p' please provide the following:
% X_collection: cell array of length Sources, containing:
% { [N x D1] ; [N x D2] ; ...} (N samples, Di features) is the training set
% of multiple feature spaces.
% t: of size C x N (C classes, N samples) is the training labels.
% standardize_flag: boolean cell array {Sources x 1} turns ON/OFF data standardization for each source
% convergence_used: values [1 or 2] is the training termination criterion (see
%                   conv.A and conv.N of theoretical background
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

function OUTPUT = train_mRVM2_MKL(input_flag,X_collection,t,standardize_flag,convergence_used,Nmax,kernel_type,kernel_param,plot_flag,dataset_name)

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
    
    Sources = input('please enter number of kernels > ');
    
    X_collection=cell(Sources,1);
    
    kernel_type = cell(Sources,1);
    kernel_param = cell(Sources,1);
    
    i=1;
    while i<=Sources
        %% LOAD DATA
        try
            fprintf(1,'\n Please provide the variable name of the %d train source',i);
            X_collection{i} = input(' > ');
            
            standardize_flag{i} = logical(input('standardize_data? (1/0) > '));
                        
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
    
end
Sources = length(X_collection);

%% INITIALIZATIONS
Xtrain_collection = cell(Sources,1);
Xtrain = [];
for s=1:Sources
    if standardize_flag{s}
    [Xtrain_collection{s}, Xmean{s}, Xstd{s}] = standardize_data(X_collection{s});
    else
        Xtrain_collection{s} = X_collection{s};
    end
    Xtrain = [Xtrain Xtrain_collection{s}];
end

N = size(Xtrain,1);
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
disp('building composite kernel...')
% BUILD COMPOSITE KERNEL
Kernels = zeros(N,N,Sources);
beta = 1/Sources * ones(1,Sources);

Ktrain = zeros(N,N);

for i=1:Sources
    Kernels(:,:,i) = ...
        build_standarized_kernel(Xtrain_collection{i},...
        Xtrain_collection{i},kernel_type{i},...
        kernel_param{i});
    Ktrain = Ktrain + beta(i) * Kernels(:,:,i);
end

%% MAIN LOOP
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
    %% MKL learning
    % Get beta for Quadratic programming-----------------
    M=zeros(N,C,Sources);
    for kernel_index=1:Sources
        M(:,:,kernel_index) = Kernels(:,:,kernel_index) * W;
    end
    
    Omega = zeros(Sources, Sources);
    for class_index = 1 : C
        Mtemp = squeeze(M(:,class_index,:));
        Omega = Omega + Mtemp'*Mtemp;
    end
    
    for kernel_index = 1 : Sources
        f(kernel_index)=sum(sum(Y'.*squeeze(M(:,:,kernel_index))));
    end
    
    beta = quadprog(Omega, -f, zeros(Sources), zeros(Sources, 1 ),...
        ones(Sources), ones(Sources, 1), zeros(Sources,1) );
    
    % Update Composite Kernel----------
    Ktrain=zeros(N,N);
    for kernel_index=1:Sources
        Ktrain=Ktrain+beta(kernel_index)*Kernels(:,:,kernel_index);
    end
    %% stop when the maximum number of iterations have been reached.
    if (convergence_used==1 && converged_loga) || (convergence_used==2 && mRVM2_iterator==Nmax)
        break;
    end
    
    % break if not converged after 10*N
    if mRVM2_iterator == 10 * N
        break;
        warning('not converged - training terminated after maximum iterations')
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

X_prototypical = cell(Sources,1);
for s=1:Sources
    X_aux = X_collection{i};
    X_prototypical{s} = X_aux(active_samples,:);
end

X_prototypical_standardized = cell(Sources,1);
for s=1:Sources
    if standardize_flag{s}
        X_aux = Xtrain_collection{s};
        X_prototypical_standardized{s} = X_aux(active_samples,:);
    else
        X_prototypical_standardized{s} = nan;
    end
end

OUTPUT = mRVM_train_output(model_used, dataset_name, N, X_prototypical,standardize_flag,X_prototypical_standardized, Xmean,Xstd, Ktrain(active_samples,active_samples),...
    W(active_samples,:), active_samples,kernel_type,kernel_param,Sources,beta);
end