function s = Bounds( s, Lb, Ub)
temp = s;
for i=1:length(s)
    if i==4%����ѧϰ�� �����Ķ�������
        temp(:,i) =temp(:,i);
    else
        temp(:,i) =round(temp(:,i));
    end
end

% �жϲ����Ƿ񳬳��趨�ķ�Χ

for i=1:length(s)
    if temp(:,i)>Ub(i) | temp(:,i)<Lb(i) 
        if i==4%����ѧϰ�� �����Ķ�������
            temp(:,i) =rand*(Ub(i)-Lb(i))+Lb(i);
        else
            temp(:,i) =round(rand*(Ub(i)-Lb(i))+Lb(i));
        end
    end
end
s = temp;