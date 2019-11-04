numHiddenUnits = 40;
sens_diff=(diff(Sen(:,:)'))';
%sens_diff((sens_diff<0.005))=0;

Pos_diff=(diff(Pos(3,:)'))';
Trans_diff=(diff(Trans(:,1:2)));
%Trans_diff(Trans_diff<0.01)=0;

% t=[Pos_diff(198:end-3-100,1:2)]';
  x=[Pos(3,30:end-2-100)];
x=Sen([1],30:end-100);
% t=Trans(2:end-100,3)';
% 
% x=[sens([1:5 ,7:8],199:end);Trans(200:end,1)'];
% t=Trans(200:end,2)';
% 
 %x=asd([1:5 ,7:8],199:end);
% %t=sens(2,199:end);
 %x=[sens(1:4,199:end)];
%x=sens_diff([2:6],29:end-100);

t=Sen([2:6],30:end-100);
%x=[Sen([1:6],30:end-100);sens_diff([1:6],29:end-100)];
%x=sens(:,1:end-100);

%x(x==NaN)=0;
%t=sens(3,2:end);


inputSize = size(x,1);
numResponses = size(t,1);

divi=floor(0.8*length(x));
numTimeStepsTrain=divi;


xm=mean(x');
%xm=x(:,1);
xs=std(x');
tm=mean(t');
ts=std(t');

% Normalizing by mean and stdev for both inputs/outputs.
for z=1:inputSize
    x(z,:)= x(z,:)-xm(z);
     x(z,:)= x(z,:)/xs(z);
end
for z=1:numResponses
        t(z,:)= t(z,:)-tm(z);
     t(z,:)= t(z,:)/ts(z);
end


  
XTrain = x(:,1:divi);
YTrain = t(:,1:divi);
XTest = x(:,divi+1:end);
YTest = t(:,divi+1:end);


layers = [ ...
    sequenceInputLayer(inputSize)
     tanhLayer
   %tanhLayer(numHiddenUnits)
    %dropoutLayer(0.5) %dropout should prevent overfitting and make predicitons more robust to noise
    %lstmLayer(numHiddenUnits)
    lstmLayer(numHiddenUnits)%,'OutputMode','sequence')
    lstmLayer(numHiddenUnits)
 
    fullyConnectedLayer(numResponses)
    regressionLayer];


opts = trainingOptions('adam', ...
    'MaxEpochs',200, ... % number of training iterations.
    'MiniBatchSize', 512,... %%%512
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005*1, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125*1, ...%%changed
    'LearnRateDropFactor',0.2/1, ...%%
    'Verbose',1, ...
    'Plots','training-progress', 'ExecutionEnvironment', 'gpu');

% Training.
[net,info] = trainNetwork(XTrain,YTrain,layers,opts);

% Predict for the whole dataset.
[net,YPred_o]= predictAndUpdateState(net,x);
%[net,YPred] = predictAndUpdateState(net,XTrain(:,end));

% Undo the normalization.

for z=1:numResponses
     t(z,:)= t(z,:)*ts(z);
     YPred_o(z,:)=YPred_o(z,:)*ts(z);
end

% Plotting.
plot(YPred_o(1,:),'r')
hold on
plot(t(1,:),'b')

len=length(t);
%err=rssq(YPred_o(:,1:len)-t(:,1:len));
err=abs(YPred_o(:,1:len)-t(:,1:len));
err_test=mean(err(divi:end))

