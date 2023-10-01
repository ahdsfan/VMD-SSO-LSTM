function y=fitness(x,p,t,pt,tt)
rng(0)
numFeatures = size(p,1);%输入节点数
numResponses = size(t,1);%输出节点数
miniBatchSize = 10; %batchsize
numHiddenUnits1 = x(1);
numHiddenUnits2 = x(2);
maxEpochs=x(3);
learning_rate=x(4);
layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits1)
    lstmLayer(numHiddenUnits2)
    fullyConnectedLayer(numResponses)
    regressionLayer];
options = trainingOptions('adam', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',learning_rate, ...
    'GradientThreshold',1, ...
    'Shuffle','every-epoch', ...
    'Verbose',false);


net = trainNetwork(p,t,layers,options);

YPred = predict(net,pt,'MiniBatchSize',1);YPred=double(YPred);
y =mse(YPred-tt);
% 以mse为适应度函数，优化算法目的就是找到一组超参数 使网络的mse最低
rng((100*sum(clock)))

