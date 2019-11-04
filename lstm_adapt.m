
 t=[Pos(1:2,199:end-2-100)];

x=Sen([1:6],199:end-100);



inputSize = size(x,1);
numResponses = size(t,1);

divi=floor(0.8*length(x));
numTimeStepsTrain=divi;


xm=mean(x');
%xm=x(:,1);
xs=1*std(x');
tm=mean(t');
ts=1*std(t');

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

%XTrain = x(:,1:15000);
%YTrain = t(:,1:15000);

XTest = x(:,divi+1:end);
YTest = t(:,divi+1:end);

opts = trainingOptions('adam', ...
    'MaxEpochs',200, ... % number of training iterations.
    'MiniBatchSize', 512,... %%%512
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005*10, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125*1, ...%%changed
    'LearnRateDropFactor',0.2/1, ...%%
    'Verbose',1, ...
    'Plots','training-progress', 'ExecutionEnvironment', 'cpu');


lgraph = layerGraph(net.Layers); 
%lgraph = removeLayers(lgraph,'dropout');
layers = lgraph.Layers;
connections = lgraph.Connections;


layers(1:3) = freezeWeights(layers(1:3));

layers(4)=fullyConnectedLayer(numResponses);
%lgraph = createLgraphUsingConnections(layers,connections);

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
net_cut=net.Layers(1:3);
    layers = [ ...
        net_cut
        regressionLayer]; % has to be a regression layer in order to be a 'valid system'
    net_new=SeriesNetwork(layers);
predict(net_new,x);



plot(YPred_o(1,:),'r')
hold on
plot(t(1,:),'b')

len=length(t);
err=rssq(YPred_o(:,1:len)-t(:,1:len));
%err=abs(YPred_o(:,1:len)-t(:,1:len));
err_test=mean(err(divi:end))

% %  
% for j=1
% % err=YPred_o(1,j:end-2)-t(1,1:end-j+1);  
%  err=YPred_o(1,1:end-2-j+1)-t(1,j:end); 
%  % plot(err)
% %  hold on
% end
% %  hold on