% authors: Ioannis Psorakis psorakis@gmail.com, Theodoros Damoulas theo@dcs.gla.ac.uk
%    Copyright NCR, 2009 
%
%    Use of this code is subject to the Terms and Conditions of the Research License Agreement,
%    agreed to when this code was downloaded, and a copy of which is available at 
%    http://www.dcs.gla.ac.uk/inference/pMKL/Terms_and_Conditions.html.

function K = kernel_function(X,Y,type,param)
%disp('---Kernel function---')

if strcmp(type,'linear')
    %check if dimensions match and perform transposing
    %     x_dims=size(x);
    %     x_lines=x_dims(1);
    %     x_cols=x_dims(2);
    %
    %     y_dims=size(y);
    %     y_lines=y_dims(1);
    %     y_cols=y_dims(2);
    
    %     if x_lines>x_cols
    %         x=x';
    %     end
    %
    %     if y_lines<y_cols
    %         y=y';
    %     end
    
    K=X*Y'; 
    
elseif strcmp(type,'gaussian')
    N = size(X,1);
    M = size(Y,1);
    D = size(X,2);
    
    if ~exist('param','var')
        param = 1/D;
    end
    
    if size(param)==ones(1,2)
        param = repmat(param,1,D);
    end
    
    T=diag(-param);% The diagonal matrix DxD with D parameters
    K=exp(repmat(diag(X*T*X'),1,M)+repmat(diag(Y*T*Y')',N,1)-2*X*T*Y');
  
elseif strcmp(type,'polynomial')
    %     if(size(x)==size(y))
    %         %         pol=1;
    %         %         for i=1:max(size(x))
    %         %             pol = pol * (1 + x(i)*y(i))^param;
    %         %         end
    %         %
    %         %         k=pol;
    %         %
    %         k = prod( ones(1,length(x(1,:))) + x.*y)^param;
    %         %  ok=1;
    %     else
    %         k=NaN;
    %         % ok=0;
    %     end
    if ~exist('param','var')
        param = 2;
    end
    
    prod = X*Y';
    K = (prod + ones(size(prod))).^param;
end
