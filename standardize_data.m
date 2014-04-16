% authors: Ioannis Psorakis psorakis@gmail.com, Theodoros Damoulas
% theo@dcs.gla.ac.uk
%    Copyright NCR, 2009
%
%    Use of this code is subject to the Terms and Conditions of the Research License Agreement,
%    agreed to when this code was downloaded, and a copy of which is available at
%    http://www.dcs.gla.ac.uk/inference/pMKL/Terms_and_Conditions.html.

function [Xs Xmean Xstd] = standardize_data(X,Xmean,Xstd)

N = size(X,1);

if ~exist('Xmean','var') && ~exist('Xstd','var')
  
    Xmean = mean(X,1);
    
    Xs = X-repmat(Xmean,N,1);
    
    Xstd = std(Xs,[],1);
    
    std_zeros = find(Xstd==0);
    
    if ~isempty(std_zeros)
        disp('Warning: non-discriminative features exist in the data:');
        fprintf('feature index: %d\n\n',std_zeros);
        Xstd(std_zeros) = 10e-6;
    end
    
    Xs =Xs./repmat(Xstd,N,1);
    
else
    Xs = X-repmat(Xmean,N,1);
    
    std_zeros = find(Xstd==0);
    
    if ~isempty(std_zeros)
        disp('Warning: non-discriminative features exist in the data:');
        fprintf('feature index: %d\n\n',std_zeros);
        Xstd(std_zeros) = 10e-6;
    end
    
    Xs =Xs./repmat(Xstd,N,1);
end