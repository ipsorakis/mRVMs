% authors: Ioannis Psorakis psorakis@gmail.com, Theodoros Damoulas theo@dcs.gla.ac.uk
%    Copyright NCR, 2009 
%
%    Use of this code is subject to the Terms and Conditions of the Research License Agreement,
%    agreed to when this code was downloaded, and a copy of which is available at 
%    http://www.dcs.gla.ac.uk/inference/pMKL/Terms_and_Conditions.html.

% W: NXC
% K: NXN
% y: 1XN
% Y: CXN
% A: diag(N)
function pw = postW_map(W,K,Y,A)

N = size(K,1);

C = size(W,2);
pw = zeros(N,C);
for i=1:C
    y=Y(i,:);
    Ac = diag(A(:,i));
    pw(:,i) = ((K*K'+ Ac)\K)*y'; 
    %pw(:,i) = chol_inv(K*K'+ Ac)*K*y';
end