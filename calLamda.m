
function [rate]=calLamda(mode,matrixSketch,topK,hasS)
[I1,I2]=size(matrixSketch);
minI=min(I1,I2);
[~,S,~]=svds(matrixSketch,minI);
hasSValue=0;
for i=1:topK
    hasSValue=hasSValue+hasS(i,i);
end
Svalue=0;
for i=1:minI
    Svalue=Svalue+S(i,i);
end
rate=hasSValue/Svalue;
end