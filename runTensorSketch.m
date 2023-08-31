function[mode1Sketch,mode2Sketch,mode3Sketch,detTime,insertTime,percentageCDFpara,partTime1,partTime2,partTime3,L1,L2,L3,indexZore1,indexZore2,indexZore3,numShrink1,numShrink2,numShrink3,projectionMatrix1,projectionMatrix2,projectionMatrix3,flagScore1,flagScore2,flagScore3,trainingScore1,trainingScore2,trainingScore3,trainingScoreSum,norScoreArr1,norScoreArr2,norScoreArr3,norScoreArr_Totally,abnorScoreArr1,abnorScoreArr2,abnorScoreArr3,abnorScoreArr_Totally,TPR,FPR,Recall,Precision,f1Score,accuracy,flagScoreWeight]=runTensorSketch(DataName,numSliceForSketchStorage,topK1,topK2,topK3,trainingSliceNum,deteionSliceNum,anomalySliceNum,mu,sigma,abnormalPointNum,weight1,weight2,weight3,seq,percentageCDFpara,path,allDataI3,ID)
[allDataI1,allDataI2,~]=size(DataName);

%Initialize the sketch of 1-mode unfolding matrix
mode1Sketch_I1=allDataI1;
mode1Sketch_I2=allDataI2*numSliceForSketchStorage;
mode1Sketch=zeros(mode1Sketch_I1,mode1Sketch_I2);%the sketch of 1-mode unfolding matrix
mode1SketchU=zeros(mode1Sketch_I1,topK1);%Orthogonal matrix of mode1Sketch
projectionMatrix1=zeros(mode1Sketch_I1,mode1Sketch_I1);%Projection matrix 


%Initialize the sketch of 2-mode unfolding matrix
mode2Sketch_I1=allDataI2;
mode2Sketch_I2=allDataI1*numSliceForSketchStorage;
mode2Sketch=zeros(mode2Sketch_I1,mode2Sketch_I2);%the sketch of 2-mode unfolding matrix
mode2SketchU=zeros(mode2Sketch_I1,topK2);%Orthogonal matrix of mode2Sketch
projectionMatrix2=zeros(mode2Sketch_I1,mode2Sketch_I1);

%Initialize the sketch of 3-mode unfolding matrix
mode3Sketch_I1=numSliceForSketchStorage;
mode3Sketch_I2=allDataI1*allDataI2;
mode3Sketch=zeros(mode3Sketch_I1,mode3Sketch_I2);%the sketch of 3-mode unfolding matrix
mode3SketchV=zeros(mode3Sketch_I2,topK3);%Orthogonal matrix of mode3Sketch
projectionMatrix3=zeros(mode3Sketch_I2,mode3Sketch_I2);



indexZore1=1;%indexZore1 represents the subscript of the zero-value column in mode1Sketch
indexZore2=1;%indexZore2 represents the subscript of the zero-value column in mode2Sketch
indexZore3=1;%indexZore3 represents the subscript of the zero-value row in mode3Sketch

numShrink1=0;%the number of shrink operation in mode1Sketch
numShrink2=0;%the number of shrink operation in mode2Sketch
numShrink3=0;%the number of shrink operation in mode3Sketch


L1=0;
L2=0;
L3=0;


partTime1=0;
partTime2=0;
partTime3=0;

anomalySlicePos=randperm(deteionSliceNum,anomalySliceNum)+trainingSliceNum;
anomalySlicePos=sort(anomalySlicePos);

TP=0;%Ture Positive 
FP=0;%False Positive
FN=0;%False Negative
TN=0;%Ture Negative

%Record the composite outlier scores during the training phase
trainingScore1=zeros(1,trainingSliceNum-numSliceForSketchStorage);
trainingScore2=zeros(1,trainingSliceNum-numSliceForSketchStorage);
trainingScore3=zeros(1,trainingSliceNum-numSliceForSketchStorage);
trainingScoreSum=zeros(1,trainingSliceNum-numSliceForSketchStorage);
trainingCount=1;

%Record the composite outlier scores of normal data during the testing phase
norScoreArr1=zeros(1,deteionSliceNum-anomalySliceNum); 
norScoreArr2=zeros(1,deteionSliceNum-anomalySliceNum);
norScoreArr3=zeros(1,deteionSliceNum-anomalySliceNum); 
norScoreArr_Totally=zeros(1,deteionSliceNum-anomalySliceNum); 
norCount=1;

%Record the composite outlier scores of abnormal data during the testing phase
abnorScoreArr1=zeros(1,anomalySliceNum);
abnorScoreArr2=zeros(1,anomalySliceNum);
abnorScoreArr3=zeros(1,anomalySliceNum);
abnorScoreArr_Totally=zeros(1,anomalySliceNum);
abnorCount=1;

%Parameter check
if(topK1>min(mode1Sketch_I1,mode1Sketch_I2) || topK2>min(mode2Sketch_I1,mode2Sketch_I2) || topK3>min(mode3Sketch_I1,mode3Sketch_I2) )
    errID = 'myComponent:inputError';
    msgtext = 'Input does not have the expected format, please check the parameter above.';
    ME = MException(errID,msgtext);
    throw(ME);
elseif(deteionSliceNum>(allDataI3-trainingSliceNum) || anomalySliceNum>deteionSliceNum)
     errID = 'myComponent:inputError';
    msgtext = 'Input does not have the expected format, please check the parameter above.';
    ME = MException(errID,msgtext);
    throw(ME);
end



%Training phase
for i=1:trainingSliceNum
    slice=DataName(:,:,i);
    if(i>numSliceForSketchStorage) 
        [curScore1]=deteAnomaly(1,slice,allDataI1,allDataI2,mode1Sketch,mode1Sketch_I1,mode1Sketch_I2,projectionMatrix1);
        [curScore2]=deteAnomaly(2,slice,allDataI1,allDataI2,mode2Sketch,mode2Sketch_I1,mode2Sketch_I2,projectionMatrix2);
        [curScore3]=deteAnomaly(3,slice,allDataI1,allDataI2,mode3Sketch,mode3Sketch_I1,mode3Sketch_I2,projectionMatrix3);
        trainingScore1(1,trainingCount)=curScore1*weight1;
        trainingScore2(1,trainingCount)=curScore2*weight2;
        trainingScore3(1,trainingCount)=curScore3*weight3;
        %Formula for calculating a composite outlier score: curScore1*weight1+curScore2*weight2+curScore3*weight3
        trainingScoreSum(1,trainingCount)=curScore1*weight1+curScore2*weight2+curScore3*weight3;
        trainingCount=trainingCount+1;
    end     
     %update the mode1Sketch by function insertByCol()
     [L1,mode1Sketch,indexZore1,mode1SketchU,projectionMatrix1,numShrink1]=insertByCol(1,slice,mode1Sketch,mode1Sketch_I1,mode1Sketch_I2,indexZore1,topK1,mode1SketchU,projectionMatrix1,numShrink1);    
     %update the mode2Sketch by function insertByCol()
     [L2,mode2Sketch,indexZore2,mode2SketchU,projectionMatrix2,numShrink2]=insertByCol(2,slice,mode2Sketch,mode2Sketch_I1,mode2Sketch_I2,indexZore2,topK2,mode2SketchU,projectionMatrix2,numShrink2);
     %update the mode3Sketch by function insertByRow()
     [L3,mode3Sketch,indexZore3,mode3SketchV,projectionMatrix3,numShrink3]=insertByRow(3,slice,allDataI1,allDataI2,mode3Sketch,mode3Sketch_I1,mode3Sketch_I2,indexZore3,topK3,mode3SketchV,projectionMatrix3,numShrink3);   
end



trainingScore1=sort(trainingScore1);
trainingScore2=sort(trainingScore2);
trainingScore3=sort(trainingScore3);
trainingScoreSum=sort(trainingScoreSum);



flagScore1=trainingScore1(1,round((trainingSliceNum-numSliceForSketchStorage)*percentageCDFpara));
flagScore2=trainingScore2(1,round((trainingSliceNum-numSliceForSketchStorage)*percentageCDFpara));
flagScore3=trainingScore3(1,round((trainingSliceNum-numSliceForSketchStorage)*percentageCDFpara)); 
%flagScoreWeight is the threshold selected based on CDF
flagScoreWeight=trainingScoreSum(1,round((trainingSliceNum-numSliceForSketchStorage)*percentageCDFpara));

timeArr1=zeros(1,deteionSliceNum);
timeArr1Count=1;
timeArr2=zeros(1,deteionSliceNum);
timeArr2Count=1;


%Testing phase
for i=1:deteionSliceNum
     slice=DataName(:,:,trainingSliceNum+i);%the current traffic matrix
     if(ismember(trainingSliceNum+i,anomalySlicePos)==1)
        outliers=normrnd(mu,sigma,[1,abnormalPointNum]);
        pos=randperm(allDataI1*allDataI2, abnormalPointNum);
        pos=sort(pos);
        for j=1:abnormalPointNum
            posX1=floor(pos(1,j)/allDataI2)+1;
            posX2=mod(pos(1,j),allDataI2);           
            if(posX2==0)
                posX1=posX1-1;
                if(posX1==0)
                    posX1=1;
                end
                posX2=allDataI2;
            end 
           slice(posX1,posX2)=slice(posX1,posX2)+outliers(1,j); 
        end
        time1=cputime;
        [curScore1]=deteAnomaly(1,slice,allDataI1,allDataI2,mode1Sketch,mode1Sketch_I1,mode1Sketch_I2,projectionMatrix1);
        [curScore2]=deteAnomaly(2,slice,allDataI1,allDataI2,mode2Sketch,mode2Sketch_I1,mode2Sketch_I2,projectionMatrix2);
        [curScore3]=deteAnomaly(3,slice,allDataI1,allDataI2,mode3Sketch,mode3Sketch_I1,mode3Sketch_I2,projectionMatrix3);
        time2=cputime;
        timeArr1(1,timeArr1Count)=time2-time1;
        timeArr1Count=timeArr1Count+1;
        curScoreWeight=curScore1*weight1+curScore2*weight2+curScore3*weight3;
        abnorScoreArr1(1,abnorCount)=curScore1*weight1;
        abnorScoreArr2(1,abnorCount)=curScore2*weight2;
        abnorScoreArr3(1,abnorCount)=curScore3*weight3;
        abnorScoreArr_Totally(1,abnorCount)=curScore1*weight1+curScore2*weight2+curScore3*weight3;
        abnorCount=abnorCount+1;
       if(curScoreWeight>flagScoreWeight)%if the socre of current traffic matrix is larger than threshold
           TP=TP+1;%the traffic matrix is marked as abnormal
       else
           FN=FN+1;%the traffic matrix is marked as normal
           time1=cputime;
           %Insert the traffic matrix into mode1Sketch, mode2Sketch, and mode3Sketch
           [L1,mode1Sketch,indexZore1,mode1SketchU,projectionMatrix1,numShrink1]=insertByCol(1,slice,mode1Sketch,mode1Sketch_I1,mode1Sketch_I2,indexZore1,topK1,mode1SketchU,projectionMatrix1,numShrink1);
           [L2,mode2Sketch,indexZore2,mode2SketchU,projectionMatrix2,numShrink2]=insertByCol(2,slice,mode2Sketch,mode2Sketch_I1,mode2Sketch_I2,indexZore2,topK2,mode2SketchU,projectionMatrix2,numShrink2);
           [L3,mode3Sketch,indexZore3,mode3SketchV,projectionMatrix3,numShrink3]=insertByRow(3,slice,allDataI1,allDataI2,mode3Sketch,mode3Sketch_I1,mode3Sketch_I2,indexZore3,topK3,mode3SketchV,projectionMatrix3,numShrink3);
           time2=cputime;
           timeArr2(1,timeArr2Count)=time2-time1;
           timeArr2Count=timeArr2Count+1;
       end       
     else       
       time1=cputime;
       [curScore1]=deteAnomaly(1,slice,allDataI1,allDataI2,mode1Sketch,mode1Sketch_I1,mode1Sketch_I2,projectionMatrix1);
       [curScore2]=deteAnomaly(2,slice,allDataI1,allDataI2,mode2Sketch,mode2Sketch_I1,mode2Sketch_I2,projectionMatrix2);
       [curScore3]=deteAnomaly(3,slice,allDataI1,allDataI2,mode3Sketch,mode3Sketch_I1,mode3Sketch_I2,projectionMatrix3);
       time2=cputime; 
       timeArr1(1,timeArr1Count)=time2-time1;      
       timeArr1Count=timeArr1Count+1;
       curScoreWeight=curScore1*weight1+curScore2*weight2+curScore3*weight3;
       norScoreArr1(1,norCount)=curScore1*weight1;
        norScoreArr2(1,norCount)=curScore2*weight2;
        norScoreArr3(1,norCount)=curScore3*weight3;
        norScoreArr_Totally(1,norCount)=curScore1*weight1+curScore2*weight2+curScore3*weight3;
        norCount=norCount+1;
       if(curScoreWeight>flagScoreWeight)   
           FP=FP+1; 
       else
           TN=TN+1;         
           time1=cputime;
           [L1,mode1Sketch,indexZore1,mode1SketchU,projectionMatrix1,numShrink1]=insertByCol(1,slice,mode1Sketch,mode1Sketch_I1,mode1Sketch_I2,indexZore1,topK1,mode1SketchU,projectionMatrix1,numShrink1);
           [L2,mode2Sketch,indexZore2,mode2SketchU,projectionMatrix2,numShrink2]=insertByCol(2,slice,mode2Sketch,mode2Sketch_I1,mode2Sketch_I2,indexZore2,topK2,mode2SketchU,projectionMatrix2,numShrink2);
           [L3,mode3Sketch,indexZore3,mode3SketchV,projectionMatrix3,numShrink3]=insertByRow(3,slice,allDataI1,allDataI2,mode3Sketch,mode3Sketch_I1,mode3Sketch_I2,indexZore3,topK3,mode3SketchV,projectionMatrix3,numShrink3);
           time2=cputime;  
           timeArr2(1,timeArr2Count)=time2-time1;
           timeArr2Count=timeArr2Count+1;
       end
     end
end

abnorScoreArr1=sort(abnorScoreArr1);
abnorScoreArr2=sort(abnorScoreArr2);
abnorScoreArr3=sort(abnorScoreArr3);

norScoreArr1=sort(norScoreArr1);
norScoreArr2=sort(norScoreArr2);
norScoreArr3=sort(norScoreArr3);

%evaluating indicator
TPR=TP/(TP+FN);
FPR=FP/(FP+TN);
Recall=TP/(TP+FN);
Precision=TP/(TP+FP);
f1Score=2*(Precision*Recall)/(Precision+Recall);
accuracy=(TP+TN)/(TP+FP+FN+TN);
detTime=sum(timeArr1);
insertTime=sum(timeArr2);

if((FP+TN)~=(deteionSliceNum-anomalySliceNum)   &&  (TP+FN)~=anomalySliceNum) 
    errID = 'myComponent:inputError';
    msgtext = 'FP,TN,TP,FN';
    ME = MException(errID,msgtext);%直接生成一个
    throw(ME);
end

end





