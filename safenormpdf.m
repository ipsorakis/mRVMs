function c = safenormpdf(x)
thresh=35;
x(x<-thresh)=-thresh;
x(x>thresh)=thresh;
c=normpdf(x);
end