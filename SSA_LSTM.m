%% ��ȸ�Ż�LSTM��Ԫ�ع�Ԥ��
clc;clear;close all;format compact
%%
data=xlsread('predict_data_zhouqi_bodong.xlsx','bodong','B2:E91') %��Ҫ��������ݼ�
x=data(:,1:4);
y=data(:,5);
method=@mapminmax;%��һ��
% method=@mapstd;%��׼��
[xs,mappingx]=method(x');x=xs';
[ys,mappingy]=method(y');y=ys';

%��������
n=size(x,1);
m=round(n*0.7);%ǰ70%ѵ�� ��30%����
XTrain=x(1:m,:)';
XTest=x(m+1:end,:)';
YTrain=y(1:m,:)';
YTest=y(m+1:end,:)';
%% ����ssa�Ż�
[x ,fit_gen,process]=ssaforlstm(XTrain,YTrain,XTest,YTest);%�ֱ��������ڵ� ѵ��������ѧϰ��Ѱ��

%% ����Ӧ������
plfit(fit_gen,'SSA')
disp('�Ż��ĳ�����Ϊ��')
disp('L1:'),x(1)
disp('L2:'),x(2)
disp('K:'),x(3)
disp('lr:'),x(4)

%% �����Ż��õ��Ĳ�������ѵ��
train=0;%�Ƿ�����ѵ��
    rng(0)
    numFeatures = size(XTrain,1);%����ڵ���
    numResponses = size(YTrain,1);%����ڵ���
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
%% Ԥ��
YPred = predict(net,XTest);
YPred=double(YPred);
% ����һ��
predict_value=method('reverse',YPred,mappingy);
true_value=method('reverse',YTest,mappingy);
%%
disp('�������')
rmse=sqrt(mean((true_value-predict_value).^2));
disp(['��������(RMSE)��',num2str(rmse)])

mae=mean(abs(true_value-predict_value));
disp(['ƽ��������MAE����',num2str(mae)])

mape=mean(abs((true_value-predict_value)./true_value));
disp(['ƽ����԰ٷ���MAPE����',num2str(mape*100),'%'])
[r2 ,rmse] =r2_rmse(true_value,predict_value);
disp(['����Ŷȣ�r2����',num2str(r2),'%'])
fprintf('\n')

%
figure
plot(true_value,'-s','Color',[0 0 255]./255,'linewidth',1,'Markersize',5,'MarkerFaceColor',[0 0 255]./255)
hold on
plot(predict_value,'-o','Color',[0 0 0]./255,'linewidth',0.8,'Markersize',4,'MarkerFaceColor',[0 0 0]./255)
legend('ʵ��ֵ','Ԥ��ֵ')
grid on
title('SSA-LSTMģ��Ԥ����')
legend('��ʵֵ','Ԥ��ֵ')
xlabel('����')
ylabel('Ԥ��ֵ')

figure
bar((predict_value - true_value))   
legend('SSA-LSTMģ�Ͳ��Լ����')
title('SSA-LSTMģ�Ͳ��Լ����')
ylabel('���','fontsize',10)
xlabel('����','fontsize',10)

