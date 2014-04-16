
function [z y] = YTruncate(F,comp,Nsamps)
% Calculate the normalising constant and the expected value of a gaussian
% with mean f and largest component comp
% SUPER-FAST Vectorised version
% Written by Dr. Simon Rogers:  www.dcs.gla.ac.uk/inference/~srogers
%-----------------------------------------
% Copyright August 2009 Dr. Simon Rogers



N = size(F,1);
K = size(F,2);
diff = repmat(F(:,comp),1,K) - F;
u = randn([N 1 Nsamps]);
q = safenormcdf(repmat(u,[1,K,1]) + repmat(diff,[1,1,Nsamps]));
pr = repmat(prod(q,2),[1 K 1]);
pr = pr ./ q;
ind = [1:K];
ind(comp) = [];
pr(:,ind,:) = pr(:,ind,:)./repmat(q(:,comp,:),[1 K-1 1]);
pr(:,ind,:) = pr(:,ind,:).*safenormpdf(repmat(u,[1 K-1 1]) + ...
   repmat(diff(:,ind),[1 1 Nsamps]));

z = reshape(mean(pr(:,comp,:),3),N,1);
y(:,ind) = -repmat(1./z,1,K-1).*...
   reshape(mean(pr(:,ind,:),3),N,K-1) + F(:,ind);
y(:,comp) = F(:,comp) + sum(F(:,ind)-y(:,ind),2);