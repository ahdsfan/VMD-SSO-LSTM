%% vmd main.m
clear
close all
clc
warning off

%% read data
data=xlsread('leiji.xlsx','Sheet1','B2:B91');  %例如：num=xlsread('demo1.xls','sheet2','B1:B20'), %读取demo1.xls文件sheet2中的B1到B20

fs=60;

len=length(data);

t = (0:len-1)/1;            %调节曲线的横坐标值

%% VMD process
alpha=2000; 
K=5; 
tau = 0;          
DC = 0;             
init = 1;     
tol = 1e-7;      

%[modes, u_hat, omega] = VMD(data, alpha, tau, K, DC, init, tol);
[modes, u_hat, omega]=VMD(data, alpha, tau, K, DC, init, tol);
% plot imf results
freqs =(0:len-1)'*fs/len;
figure('Units','normalized','Position',[0.1, 0.1, 0.8, 0.8])
subplot(size(modes,1)+1,2,1);  %创建画布格子subplot(x,y,1)表示x行，y列，第一个
plot(t,data,'k');       %创建2维画布 x=t y=data 线形（linespec）='k'=黑色
grid on;                %显示网格线
ylabel('初始信号')      %y轴标签
title('the time domain of Variational modal decomposition (VMD)')  %变分模态分解（VMD）的时域
subplot(size(modes,1)+1,2,2);   %第二个图
[cc,y_f]=plot_fft(data,fs,1);   %调用的matlab中自带的画fft的函数当style=1,画幅值谱；function [cc,y_f]=plot_fft(y,fs,style,varargin)
plot(y_f,cc,'k');grid on;
ylabel('初始信号')
title('the spectral plot of Variational modal decomposition (VMD)')  %变分模态分解(VMD)谱图
for i = 2:size(modes,1)+1       % m = size(X,dim);%返回矩阵的行数或列数，dim=1，则返回行数;dim=2，则返回列数
   subplot(size(modes,1)+1,2,i*2-1);
    plot(t,modes(i-1,:),'k');grid on;
    ylabel(['IMF', num2str(i - 1)])
    subplot(size(modes,1)+2,2,i*2);
    [cc,y_f]=plot_fft(modes(i - 1, :),fs,1);
    plot(y_f,cc,'k');grid on;
    ylabel(['IMF', num2str(i - 1)])
end
%{
figure
modesplus1=modes(3,:)+modes(4,:)+modes(5,:)  %将分解得到的后面四个模型进行求和得到新的一个位移数据
plot(t,modesplus1)                                      %输出新的位移数据的图像
%{
figure
modesplus2=modes(1,:)+modes(2,:)  %将分解得到的后面四个模型进行求和得到新的一个位移数据
plot(t,modesplus2)  
save var     %保存变量






















