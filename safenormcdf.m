function c = safenormcdf(x)
thresh=-30;
x(x<thresh)=thresh;
c=normcdf(x);