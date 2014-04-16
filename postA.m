% authors: Ioannis Psorakis psorakis@gmail.com, Theodoros Damoulas theo@dcs.gla.ac.uk
%    Copyright NCR, 2009 
%
%    Use of this code is subject to the Terms and Conditions of the Research License Agreement,
%    agreed to when this code was downloaded, and a copy of which is available at 
%    http://www.dcs.gla.ac.uk/inference/pMKL/Terms_and_Conditions.html.

function pa = postA(W,tau,u,param)

N = size(W,1);
C = size(W,2);

pa=zeros(size(W));
T = repmat(tau,size(W));
W2 = W.^2;
U = repmat(u,size(W));

if strcmp(param,'mode')
    % WARNING!!!!!!!!!!
    % Mode is ONLY for k>1
    % ====================
    pa = (T-repmat(1/2,size(W))) ./ (W2./2 + U);
    
elseif strcmp(param,'mean')
    pa = (2*T + ones(size(W))) ./ (W2 + 2*U);
    
end

threshold = 10^6 * ones(N,C);
pa(pa>threshold) = threshold(pa>threshold);
