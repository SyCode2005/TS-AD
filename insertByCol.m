%insert the data into mode1Sketch (when the value of mode is 1-mode) or mode2Sketch  (when the value of mode is 2-mode) 
function [L,matrixSketch,indexZore,matrixSkethU,projectionMatrix,numShrink]=insertByCol(mode,slice,matrixSketch,matrixSketchI1,matrixSketchI2,indexZore,topK,matrixSkethU,projectionMatrix,numShrink)
L=round(topK/2); %L represents the level of shrink 
testSlice=slice;
if(mode==2)
    testSlice=slice';
end
[~,sliceI2]=size(testSlice);
%Update in column way
for i=1:sliceI2
    matrixSketch(:,indexZore)=testSlice(:,i);
    indexZore=indexZore+1;
    if(indexZore>matrixSketchI2)%If mode1Sketch/mode2Sketch has no zero value column
        [U,S,~]=svds(matrixSketch,topK);
        numShrink=numShrink+1;
        newS=shrinkOperationCol(S,L,topK,matrixSketchI1,matrixSketchI2);
        matrixSketch=U*newS;
        matrixSkethU=U;
        projectionMatrix=matrixSkethU*matrixSkethU';
        indexZore=L;
    end
end
end