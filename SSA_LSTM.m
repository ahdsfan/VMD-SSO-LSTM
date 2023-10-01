%% 麻雀优化LSTM多元回归预测
clc;clear;close all;format compact
%%
data=xlsread('predict_data_zhouqi_bodong.xlsx','bodong','B2:E91') %需要处理的数据集
x=data(:,1:4);
y=data(:,5);
method=@mapminmax;%归一化
% method=@mapstd;%标准化
[xs,mappingx]=method(x');x=xs';
[ys,mappingy]=method(y');y=ys';

%划分数据
n=size(x,1);
m=round(n*0.7);%前70%训练 后30%测试
XTrain=x(1:m,:)';
XTest=x(m+1:end,:)';
YTrain=y(1:m,:)';
YTest=y(m+1:end,:)';
%% 采用ssa优化
[x ,fit_gen,process]=ssaforlstm(XTrain,YTrain,XTest,YTest);%分别对隐含层节点 训练次数与学习率寻优

%% 画适应度曲线
plfit(fit_gen,'SSA')
disp('优化的超参数为：')
disp('L1:'),x(1)
disp('L2:'),x(2)
disp('K:'),x(3)
disp('lr:'),x(4)

%% 利用优化得到的参数重新训练
train=0;%是否重新训练
    rng(0)
    numFeatures = size(XTrain,1);%输入节点数
    numResponses = size(YTrain,1);%输出节点数
    miniBatchSize = 20; %batchsize
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
        'Verbose',true,...
        'Plots','training-progress');

net = trainNetwork(XTrain,YTrain,layers,options);
%% 预测
YPred = predict(net,XTest);
YPred=double(YPred);
% 反归一化
predict_value=method('reverse',YPred,mappingy);
true_value=method('reverse',YTest,mappingy);
%%
disp('结果分析')
rmse=sqrt(mean((true_value-predict_value).^2));
disp(['根均方差(RMSE)：',num2str(rmse)])

mae=mean(abs(true_value-predict_value));
disp(['平均绝对误差（MAE）：',num2str(mae)])

mape=mean(abs((true_value-predict_value)./true_value));
disp(['平均相对百分误差（MAPE）：',num2str(mape*100),'%'])
[r2 ,rmse] =r2_rmse(true_value,predict_value);
disp(['拟合优度（r2）：',num2str(r2),'%'])
fprintf('\n')

%
figure
plot(true_value,'-s','Color',[0 0 255]./255,'linewidth',1,'Markersize',5,'MarkerFaceColor',[0 0 255]./255)
hold on
plot(predict_value,'-o','Color',[0 0 0]./255,'linewidth',0.8,'Markersize',4,'MarkerFaceColor',[0 0 0]./255)
legend('实际值','预测值')
grid on
title('SSA-LSTM模型预测结果')
legend('真实值','预测值')
xlabel('样本')
ylabel('预测值')

figure
bar((predict_value - true_value))   
legend('SSA-LSTM模型测试集误差')
title('SSA-LSTM模型测试集误差')
ylabel('误差','fontsize',10)
xlabel('样本','fontsize',10)

