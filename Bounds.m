function s = Bounds( s, Lb, Ub)
temp = s;
for i=1:length(s)
    if i==4%除了学习率 其他的都是整数
        temp(:,i) =temp(:,i);
    else
        temp(:,i) =round(temp(:,i));
    end
end

% 判断参数是否超出设定的范围

for i=1:length(s)
    if temp(:,i)>Ub(i) | temp(:,i)<Lb(i) 
        if i==4%除了学习率 其他的都是整数
            temp(:,i) =rand*(Ub(i)-Lb(i))+Lb(i);
        else
            temp(:,i) =round(rand*(Ub(i)-Lb(i))+Lb(i));
        end
    end
end
s = temp;