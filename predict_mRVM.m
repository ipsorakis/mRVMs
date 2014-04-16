%% mRVM1
% authors: Ioannis Psorakis psorakis@gmail.com, Theodoros Damoulas theo@dcs.gla.ac.uk
%    Copyright NCR, 2009
%performs predictions given the results of the training phase, described by
%TRAIN_OBJECT (see train_mRVM1) and a collection of test samples Xtest. In
%case the test_labels are available, it also performs a calculation of the
%class recognition accuracy of the algorithm.
%INPUTS:
% TRAIN_OBJECT: the output of train_mRVM1. Contains the relevance vectors, training kernel, regressors W etc.
% Xtest: the test data of size Ntest x D where D the number of features SAME with the feature number of Xtrain.
% test_labels: optional variable or size C x Ntest where C the number of
% classes.
%OUTPUTS:
% class_membership_probabilities: of size Ntest x C, where the element
%               (i,c) describes the degree of belief that sample-i belongs to class-c.
% class_recognition_accuracy: if test_labels are provided, tests the
%                   accuracy of the algorithm based on the values of
%                   class_membership_probabilities

function [class_membership_probabilities class_recognition_accuracy] = predict_mRVM(TRAIN_OBJECT,Xtest,test_labels)

if TRAIN_OBJECT.standardize_flag
    % standardize test set
    Xtest = standardize_data(Xtest,TRAIN_OBJECT.Xmean,TRAIN_OBJECT.Xstd);
    % test kernel
    Ktest = build_standarized_kernel...
        (Xtest,TRAIN_OBJECT.X_prototypical_standardized,TRAIN_OBJECT.kernel_type,TRAIN_OBJECT.kernel_param);
else
    Ktest = build_standarized_kernel...
        (Xtest,TRAIN_OBJECT.X_prototypical,TRAIN_OBJECT.kernel_type,TRAIN_OBJECT.kernel_param);
end

%% Run multinomial probit likelihood to retrieve class membership probabilities
% for each test sample the class membership distribution Ntest X C is
class_membership_probabilities = t_likelihood(TRAIN_OBJECT.W_sparse,Ktest);
%% Class recognition accuracy if test_labels exist

if exist('test_labels','var')
    class_recognition_accuracy = find_accuracy(class_membership_probabilities,test_labels');
else
    class_recognition_accuracy = nan;
end
end

%% AUXILIARY FUNCTIONS
% authors: Ioannis Psorakis psorakis@gmail.com, Theodoros Damoulas
% theo@dcs.gla.ac.uk
%    Copyright NCR, 2009
%
%    Use of this code is subject to the Terms and Conditions of the Research License Agreement,
%    agreed to when this code was downloaded, and a copy of which is available at
%    http://www.dcs.gla.ac.uk/inference/pMKL/Terms_and_Conditions.html.


function result= t_likelihood(W,K)
C = size(W,2);

result = GHermiteMProbit(20,W,K);
%% normalize probabilities
result = result ./ repmat( sum(result,2),1,C);
end

function FVal = GHermiteMProbit(npt,W,K)
%---------------------------------------------------%
% Inputs:                                           %
%  npt - Number of quadrature points for estimation %
%  W - The Regressors for the multinomial Probit    %
%      (NxC) with N the no. of training samples and %
%      C the no. of classes.                        %
%  K - The Kernel (NxN if training NtestxN if test  %
%      with Ntest-no. of test samples               %
%                                                   %
% Outputs:                                          %
%  FVal - An NxC matrix with multinomial probit     %
%         class probabilities (normalized).         %
%                                                   %
%---------------------------------------------------%
% Values for the roots xpt and for the weights wt   %
% are copied from S Bocquet file GaussHermite.      %
%---------------------------------------------------%
% T. Damoulas, July 2009

% authors: Ioannis Psorakis psorakis@gmail.com, Theodoros Damoulas theo@dcs.gla.ac.uk
%    Copyright NCR, 2009
%
%    Use of this code is subject to the Terms and Conditions of the Research License Agreement,
%    agreed to when this code was downloaded, and a copy of which is available at
%    http://www.dcs.gla.ac.uk/inference/pMKL/Terms_and_Conditions.html.

[~,C]=size(W);
N=size(K,1);

persistent npts ipos wt xpt

xpt(1:39) =[ 0.707106781186548,0.524647623275290, 1.650680123885785,0.436077411927617, 1.335849074013697, 2.350604973674492,0.381186990207322, 1.157193712446780, 1.981656756695843,2.930637420257244,0.342901327223705, 1.036610829789514, 1.756683649299882,2.532731674232790, 3.436159118837738,0.314240376254359, 0.947788391240164, 1.597682635152605,2.279507080501060, 3.020637025120890, 3.889724897869782,0.27348104613815,  0.82295144914466, 1.38025853919888,1.95178799091625,  2.54620215784748, 3.17699916197996,3.86944790486012,  4.68873893930582,0.2453407083009,   0.7374737285454, 1.2340762153953,1.7385377121166,   2.2549740020893, 2.7888060584281,3.3478545673832,   3.9447640401156, 4.6036824495507,5.3874808900112 ];
wt(1:39) =[ 8.862269254528d-1,8.049140900055d-1, 8.131283544725d-2,7.246295952244d-1, 1.570673203229d-1, 4.530009905509d-3,6.611470125582d-1, 2.078023258149d-1, 1.707798300741d-2,1.996040722114d-4,6.108626337353d-1, 2.401386110823d-1, 3.387439445548d-2,1.343645746781d-3, 7.640432855233d-6,5.701352362625d-1, 2.604923102642d-1, 5.160798561588d-2,3.905390584629d-3, 8.573687043588d-5, 2.658551684356d-7,5.079294790166d-1, 2.806474585285d-1, 8.381004139899d-2,1.288031153551d-2, 9.322840086242d-4, 2.711860092538d-5,2.320980844865d-7, 2.654807474011d-10,4.622436696006d-1, 2.866755053628d-1, 1.090172060200d-1,2.481052088746d-2, 3.243773342238d-3, 2.283386360163d-4,7.802556478532d-6, 1.086069370769d-7, 4.399340992273d-10,2.229393645534d-13 ];
npts(1:8) = int8([ 2, 4, 6, 8, 10, 12, 16, 20 ]);
ipos(1:9) = int8([ 1, 2, 4, 7, 11, 16, 22, 30, 40 ]);

%       CHECK FOR PERMISSIBLE VALUE OF NPT.

j = find(npt==npts);
if isempty(j)
    error('GaussHermite:InvalidNpt','Invalid number of points: npt must be 2,4,6,8,10,12,16 or 20')
end

%       EVALUATE SUM OF WT(I) * FUNC(X(I))

i = ipos(j):ipos(j+1)-1;

wk=(K*W)';  %% CxN (or CxNtest)

roots=[xpt(i) -xpt(i)];
R=length(roots);
Rmat=[];
for r=1:R
    Rmat = [Rmat;repmat(roots(r),N*C,C-1)];
end


M = repmat(reshape(wk',1,C*N),C,1)-repmat(wk,1,C);

%Alternative B way to avoid bug when entries zero beside ones that should be

B=reshape(M',N,C*C);

Zindex=1:C+1:C^2;
A=[];
for c=1:C
    Allindex=c:C:C^2;
    Allindex(Allindex==Zindex)=[];
    A=[A; B(:,Allindex)];
end

%B = reshape(nonzeros(M),C-1,N*C); %%

Expansion = repmat(A,2*length(i),1)+ Rmat;
Funct = prod(safenormcdf(Expansion),2); % 1 x N*C*Roots

probit=reshape(Funct,N*C,R);

P1=zeros(N*C,R/2);
		   
for l=1:R/2
	P1(:,l) = probit(:,l)+probit(:,end+1-l);
end

FVal = reshape(P1*wt(i)',N,C);
end

% authors: Ioannis Psorakis psorakis@gmail.com, Theodoros Damoulas theo@dcs.gla.ac.uk
%    Copyright NCR, 2009
%
%    Use of this code is subject to the Terms and Conditions of the Research License Agreement,
%    agreed to when this code was downloaded, and a copy of which is available at
%    http://www.dcs.gla.ac.uk/inference/pMKL/Terms_and_Conditions.html.

function y = find_accuracy(results,t)

elements = size(results,1);
maxtable = reshape(results(:)==repmat(max(results')',size(results,2),1),size(results));

tmp = maxtable + t;
sames=length(find(ismember(tmp,2)));

y=sames/elements;
end