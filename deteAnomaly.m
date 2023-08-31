%Formula for calculating outlier scores
function [curScore]=deteAnomaly(mode,slice,sliceI1,sliceI2,matrixSketch,matrixFoldI1,matrixFoldI2,projectionMatrix)
if((mode==1)|| (mode==2))        
    if(mode==2)
        slice=slice';
    end
    expression=slice-projectionMatrix*slice;
    curScore=norm(expression,'fro');
elseif(mode==3)
    vecSlice=reshape(slice,sliceI1*sliceI2,1);
    expression=(vecSlice-projectionMatrix*vecSlice);
    curScore=norm(expression,2);
end
end