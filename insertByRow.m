%insert the data into mode3Sketch (when the value of mode is 3-mode) 
function [L,matrixSketch,indexZore,modeSketchV,projectionMatrix,numShrink]=insertByRow(mode,slice,sliceI1,sliceI2,matrixSketch,matrixSketchI1,matrixSketchI2,indexZore,topK,modeSketchV,projectionMatrix,numShrink)
L=round(topK/2);
if(mode==1 || mode==2)
    errID = 'myComponent:inputError';
    msgtext = 'Input does not have the expected format.';
    ME = MException(errID,msgtext);
    throw(ME);
end
tempV=reshape(slice,1,sliceI1*sliceI2);
matrixSketch(indexZore,:)=tempV;
indexZore=indexZore+1;
if(indexZore>matrixSketchI1)
    [~,S,V]=svds(matrixSketch,topK);
    numShrink=numShrink+1;
    [newS]=shrinkOperationRow(S,L,topK,matrixSketchI1,matrixSketchI2);
    matrixSketch=newS*V';
    modeSketchV=V;
    projectionMatrix=modeSketchV*modeSketchV';
    indexZore=L;
end
end