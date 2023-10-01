function plfit(fitness,type)
figure
plot(fitness,'-s','Color',[60 180 0]./255,'linewidth',1,'Markersize',5,'MarkerFaceColor',[60 180 0]./255)
grid on
title([type,'的适应度曲线'])
xlabel('迭代次数/次')
ylabel('适应度值/MSE')