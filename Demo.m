clear;
path='AbileneDataNorm.mat';
load(path);
%The dataset has been normalized
DataName=AbileneDataNorm1;
[allDataI1,allDataI2,allDataI3]=size(DataName);
allDataI3=15000;
%Parameter settings
numSliceForSketchStorageArr=[125];%Set the memory for tensor sketch, corresponding to the parameter M in the paper
[~,numSliceForSketchStorageSize]=size(numSliceForSketchStorageArr);
%Set different rank values
topK1Arr=[10];
topK2Arr=[10];
topK3Arr=[100];
[~,topK1ArrSize]=size(topK1Arr);
[~,topK2ArrSize]=size(topK2Arr);
[~,topK3ArrSize]=size(topK3Arr);
%Number of traffic matrices required to set training thresholds
trainingSliceNumArr=[672]; 
[~,trainingSliceNumArrSize]=size(trainingSliceNumArr);
deteionSliceNumArr=zeros(1,trainingSliceNumArrSize);
anomalySliceNumArr=zeros(1,trainingSliceNumArrSize);

%deteionSliceNum: Set the number of traffic matrices required for anomaly testing
%anomalySliceNum: Set the number of traffic matrices injected with outliers in anomaly testing
for i=1:trainingSliceNumArrSize
    deteionSliceNumArr(1,i)=allDataI3-trainingSliceNumArr(1,i);
    anomalySliceNumArr(1,i)=round(deteionSliceNumArr(1,i)*0.5);
end

%Set the mean of anomalies, corresponding to parameter \mu
muArr=[0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01];
[~,numMuArr]=size(muArr);
%Set the sigma of anomalies, corresponding to parameter \sigma
sigmaArr=[0.005];
[~,numSigmaArr]=size(sigmaArr);
%Set threshold, for example: "1" represents the maximum anomaly score obtained during the training threshold stage
percentageCDFparaArr=[1];
[~,percentageCDFparaArrSize]=size(percentageCDFparaArr);
perAno=0.1;
abnormalPointNum=round(allDataI1*allDataI2*perAno);

%Set a group of weights
weight1=62.87554178;
weight2=74.20932528;
weight3=142.2195223;


numRepeat=1;
partTimeArr1=zeros(1,numRepeat);
partTimeArr2=zeros(1,numRepeat);
partTimeArr3=zeros(1,numRepeat);
for i=1:numSliceForSketchStorageSize
    numSliceForSketchStorage=numSliceForSketchStorageArr(1,i);
    for j=1:trainingSliceNumArrSize
        trainingSliceNum=trainingSliceNumArr(1,j);
        deteionSliceNum=deteionSliceNumArr(1,j);
        anomalySliceNum=anomalySliceNumArr(1,j);
        for w=1:topK3ArrSize
            topK3=topK3Arr(1,w);
            for aa=1:topK1ArrSize
                topK1=topK1Arr(1,aa);
                for bb=1:topK2ArrSize
                    topK2=topK2Arr(1,bb);
                    for p=1:percentageCDFparaArrSize
                        percentageCDFpara=percentageCDFparaArr(1,p);
                        ID=randperm(10000000,1);
                        for v=1:numSigmaArr 
                            sigma=sigmaArr(1,v); 
                            for k=1:numMuArr
                                mu=muArr(1,k); 
                                RecallArr=zeros(1,numRepeat);
                                PrecisionArr=zeros(1,numRepeat);
                                f1ScoreArr=zeros(1,numRepeat);
                                accuracyArr=zeros(1,numRepeat);
                                for o=1:numRepeat
                                    seq=o;
                                    [mode1Sketch,mode2Sketch,mode3Sketch,detTime,insertTime,percentageCDFpara,partTime1,partTime2,partTime3,L1,L2,L3,indexZore1,indexZore2,indexZore3,numShrink1,numShrink2,numShrink3,projectionMatrix1,projectionMatrix2,projectionMatrix3,flagScore1,flagScore2,flagScore3,trainingScore1,trainingScore2,trainingScore3,trainingScoreSum,norScoreArr1,norScoreArr2,norScoreArr3,norScoreArr_Totally,abnorScoreArr1,abnorScoreArr2,abnorScoreArr3,abnorScoreArr_Totally,TPR,FPR,Recall,Precision,f1Score,accuracy,flagScoreWeight]=runTensorSketch(DataName,numSliceForSketchStorage,topK1,topK2,topK3,trainingSliceNum,deteionSliceNum,anomalySliceNum,mu,sigma,abnormalPointNum,weight1,weight2,weight3,seq,percentageCDFpara,path,allDataI3,ID);                                
                                    partTimeArr1(1,o)=detTime;
                                    partTimeArr2(1,o)=insertTime;
                                    RecallArr(1,o)=Recall;
                                    PrecisionArr(1,o)=Precision;
                                    f1ScoreArr(1,o)=f1Score;
                                    accuracyArr(1,o)=accuracy;
                                end
                                averRecall=(sum(RecallArr)/numRepeat);
                                averPrecision=(sum(PrecisionArr)/numRepeat);
                                averf1Score=(sum(f1ScoreArr)/numRepeat);
                                averAccuracy=(sum(accuracyArr)/numRepeat);
                                averDetTime=(sum(partTimeArr1)/numRepeat);
                                averInsertTime=(sum(partTimeArr2)/numRepeat);
                                disp("mu="+mu+",sigma="+sigma+",averRecall="+averRecall+",averPrecision="+averPrecision+",averf1Score="+averf1Score+",averAccuracy="+averAccuracy+",flagScoreWeight="+flagScoreWeight+",topK1="+topK1+",topK2="+topK2+",topK3="+topK3+",percentageCDFpara="+percentageCDFpara+",trainingSliceNum="+trainingSliceNum+",numSliceForSketchStorage="+numSliceForSketchStorage+",flagScoreWeight="+flagScoreWeight+",ID="+ID+",allDataI3="+allDataI3+",perAno="+perAno+",aver DetTime="+averDetTime+",aver InsertTime="+averInsertTime);
                            end
                            disp(" ");
                        end
                    end
                end
            end
        end
    end
end
