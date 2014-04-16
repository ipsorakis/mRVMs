% authors: Ioannis Psorakis psorakis@gmail.com, Theodoros Damoulas theo@dcs.gla.ac.uk
%    Copyright NCR, 2009 
%
%    Use of this code is subject to the Terms and Conditions of the Research License Agreement,
%    agreed to when this code was downloaded, and a copy of which is available at 
%    http://www.dcs.gla.ac.uk/inference/pMKL/Terms_and_Conditions.html.

function X = chol_inv(K,B)

% Thanks to Theo, Keith, Vassilis
% solve_* are from lightspeed toolbox
% inv(K)*B = X

if ~exist('B','var')
    B = eye(size(K,1));
end

try 
    L=chol(K,'lower');
    y=solve_tril(L, B);
    X=solve_triu(L', y); 
catch ME
    ME.stack;
    X = K\B;
end