function [bestX,Convergence_curve,result]=ssaforlstm(P_train,T_train,P_test,T_test)
%% ��������
pop=5; % ��Ⱥ��
M=10; % ����������
dim=4;%һ����4��������Ҫ�Ż�
lb=[1   1   1  0.0001];%�ֱ������lstm������ڵ� ѵ��������ѧϰ��Ѱ��
ub=[200 200 500  0.01];%����ֱ����4�����������½磬�����һ�������ķ�Χ����1-100

P_percent = 0.2;    %producers ��ȫ����Ⱥ��ռ��
pNum = round( pop *  P_percent );    %  producers������

%��ʼ����Ⱥ
for i = 1 : pop
    for j=1:dim
        if j==4%����ѧϰ�� �����Ķ�������
            x( i, j ) = (ub(j)-lb(j))*rand+lb(j);
        else
            x( i, j ) = round((ub(j)-lb(j))*rand+lb(j));
        end
    end
    fit( i )=fitness(x(i,:),P_train,T_train,P_test,T_test);
end
pFit = fit;
pX = x;
fMin=fit(1);
bestX = x( i, : );

for t = 1 : M
    
    [ ~, sortIndex ] = sort( pFit );% Sort.��С����
    [fmax,B]=max( pFit );
    worse= x(B,:);
    r2=rand(1);
    %%%%%%%%%%%%%5%%%%%%��һ��λΪ�����ߣ�̽���ߣ���λ�ø���%%%%%%%%%%%%%%%%%%%%%%%%%
    if(r2<0.8)%Ԥ��ֵ��С��˵��û�в�ʳ�߳���
        for i = 1 : pNum  %r2С��0.8�ķ����ߵĸı䣨1-20��                                                 % Equation (3)
            r1=rand(1);
            x( sortIndex( i ), : ) = pX( sortIndex( i ), : )*exp(-(i)/(r1*M));%���Ա�����һ������任
            x( sortIndex( i ), : ) = Bounds( x( sortIndex( i ), : ), lb, ub );%�Գ����߽�ı�������ȥ��
            
            fit(  sortIndex( i ) )=fitness(x(sortIndex( i ),:),P_train,T_train,P_test,T_test);
        end
    else   %Ԥ��ֵ�ϴ�˵���в�ʳ�߳�����в������Ⱥ�İ�ȫ����Ҫȥ�����ط���ʳ
        for i = 1 : pNum   %r2����0.8�ķ����ߵĸı�
            x( sortIndex( i ), : ) = pX( sortIndex( i ), : )+randn(1)*ones(1,dim);
            x( sortIndex( i ), : ) = Bounds( x( sortIndex( i ), : ), lb, ub );
            
            fit(  sortIndex( i ) )=fitness(x(sortIndex( i ),:),P_train,T_train,P_test,T_test);
            
        end
        
    end
    [ ~, bestII ] = min( fit );
    bestXX = x( bestII, : );
    %%%%%%%%%%%%%5%%%%%%��һ��λΪ�����ߣ�׷���ߣ���λ�ø���%%%%%%%%%%%%%%%%%%%%%%%%%
    for i = ( pNum + 1 ) : pop     %ʣ�µĸ���ı任                % Equation (4)
        %         i
        %         sortIndex( i )
        A=floor(rand(1,dim)*2)*2-1;
        if( i>(pop/2))%��������ⲿ����ȸ����ʮ�ּ�����״̬����Ϊ���ǵ������ܵͣ�Ҳ������Ӧ��ֵ�ܲ����Ҫ�������ط���ʳ
            x( sortIndex(i ), : )=randn(1,dim).*exp((worse-pX( sortIndex( i ), : ))/(i)^2);
        else%��һ����׷������Χ����õķ�������Χ������ʳ�����Ҳ�п��ܷ���ʳ������ᣬʹ���Լ����������
            x( sortIndex( i ), : )=bestXX+(abs(( pX( sortIndex( i ), : )-bestXX)))*(A'*(A*A')^(-1))*ones(1,dim);
        end
        x( sortIndex( i ), : ) = Bounds( x( sortIndex( i ), : ), lb, ub );%�жϱ߽��Ƿ񳬳�
        fit(  sortIndex( i ) )=fitness(x(sortIndex( i ),:),P_train,T_train,P_test,T_test);
    end
    %%%%%%%%%%%%%5%%%%%%��һ��λΪ��ʶ��Σ�գ�ע������ֻ����ʶ����Σ�գ�����������������Ĳ�ʳ�ߣ�����ȸ��λ�ø���%%%%%%%%%%%%%%%%%%%%%%%%%
    c=randperm(numel(sortIndex));%%%%%%%%%���������������Ⱥ�����������λ�ã�Ҳ�����ⲿ�ֵ���ȸλ��һ��ʼ������ģ���ʶ��Σ����Ҫ����λ���ƶ���
    %������Ⱥ��Χ����ȸ��ȫ����£��������Ⱥ���ĵ���ȸ����������Կ��������ȸ��
    b=sortIndex(c(1:3));
    for j =  1  : length(b)      % Equation (5)
        if( pFit( sortIndex( b(j) ) )>(fMin) ) %������Ⱥ��Χ����ȸ��λ�øı�
            x( sortIndex( b(j) ), : )=bestX+(randn(1,dim)).*(abs(( pX( sortIndex( b(j) ), : ) -bestX)));
        else
            %������Ⱥ���ĵ���ȸ��λ�øı�
            x( sortIndex( b(j) ), : ) =pX( sortIndex( b(j) ), : )+(2*rand(1)-1)*(abs(pX( sortIndex( b(j) ), : )-worse))/ ( pFit( sortIndex( b(j) ) )-fmax+1e-50);
        end
        x( sortIndex(b(j) ), : ) = Bounds( x( sortIndex(b(j) ), : ), lb, ub );
        fit(  sortIndex( b(j)  ) )=fitness(x(sortIndex( b(j) ),:),P_train,T_train,P_test,T_test);
        
    end
    for i = 1 : pop
        if ( fit( i ) < pFit( i ) )
            pFit( i ) = fit( i );
            pX( i, : ) = x( i, : );
        end
        if( pFit( i ) < fMin )
            fMin= pFit( i );
            bestX = pX( i, : );
        end
    end
    t,fMin,bestX
    Convergence_curve(t)=fMin;
    result(t,:)=bestX;
end