%Reduce mode1Sketch/mode2Sketch to have zero value columns again
function [newS]=shrinkOperationCol(S,L,topK,matrixSketchI1,matrixSketchI2)
if(L>topK)
    errID = 'myComponent:inputError';
    msgtext = 'Input does not have the expected format.';
    ME = MException(errID,msgtext);
    throw(ME);
end
for i=1:topK
    S(i,i)=S(i,i)^2;
end
value=S(L,L);
cutS=zeros(topK,topK);
for i=1:topK
    cutS(i,i)=value;
end
temp=max(S-cutS,0);
afterShrinkS=sqrt(temp);
newS=zeros(topK,matrixSketchI2);
if(matrixSketchI2>topK)
    temp2=zeros(topK,(matrixSketchI2-topK));
    newS=[afterShrinkS,temp2];%Æ´½Ó
elseif(matrixSketchI2==topK)
      newS=afterShrinkS;   
elseif(matrixSketchI2<topK)
      errID = 'myComponent:inputError';
      msgtext = 'Input does not have the expected format.';
      ME = MException(errID,msgtext);
      throw(ME);  
end
end