function plfit(fitness,type)
figure
plot(fitness,'-s','Color',[60 180 0]./255,'linewidth',1,'Markersize',5,'MarkerFaceColor',[60 180 0]./255)
grid on
title([type,'����Ӧ������'])
xlabel('��������/��')
ylabel('��Ӧ��ֵ/MSE')