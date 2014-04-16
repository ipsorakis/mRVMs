% authors: Ioannis Psorakis psorakis@gmail.com, Theodoros Damoulas
% theo@dcs.gla.ac.uk
%    Copyright NCR, 2009 
%
%    Use of this code is subject to the Terms and Conditions of the Research License Agreement,
%    agreed to when this code was downloaded, and a copy of which is available at 
%    http://www.dcs.gla.ac.uk/inference/pMKL/Terms_and_Conditions.html.



% UPDATES Y BASED ON MAP
% Y: CXN
% t: CXN
% W: NXC
% K: NXN

function Yn = updateY(Y,t,W,K)

C = length(Y(:,1));

Y = W'*K;
% for i=1:N
%     class = find(t(:,i));
%     for j=1:C
%         if j~=class
%             if Y(j,i)>=Y(class,i)
%                 Y(j,i)=Y(class,i);
%             end
%         end 
%     end 
% end
% 
% Yn=Y;

maxes = Y(logical(t));
maxes = repmat(maxes',C,1);

Y(Y>maxes) = maxes(Y>maxes);

Yn=Y;