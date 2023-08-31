%Reduce mode3Sketchto have zero value rows again
function [newS]=shrinkOperationRow(S,L,topK,matrixSketchI1,matrixSketchI2)
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
newS=zeros(matrixSketchI1,topK);
if(matrixSketchI1>topK)
    temp2=zeros((matrixSketchI1-topK),topK);
    newS=[afterShrinkS;temp2];
elseif(matrixSketchI1==topK)
      newS=afterShrinkS;   
elseif(matrixSketchI1<topK)
      errID = 'myComponent:inputError';
      msgtext = 'Input does not have the expected format.';
      ME = MException(errID,msgtext);
      throw(ME);  
end
end