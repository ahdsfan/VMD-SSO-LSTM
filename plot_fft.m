function [cc,y_f]=plot_fft(y,fs,style,varargin)
%��style=1,����ֵ�ף���style=2,��������;��style=�����ģ���ô����ֵ�׺͹�����
%��style=1ʱ�������Զ�����2����ѡ����
%��ѡ�������������������Ҫ�鿴��Ƶ�ʶε�
%��һ������Ҫ�鿴��Ƶ�ʶ����
%�ڶ�������Ҫ�鿴��Ƶ�ʶε��յ�
%����style���߱���ѡ���������������뷢��λ�ô���
nfft= 2^nextpow2(length(y));%�ҳ�����y�ĸ���������2��ָ��ֵ���Զ��������FFT����nfft��
%nfft=1024;%��Ϊ����FFT�Ĳ���nfft
  y=y-mean(y);%ȥ��ֱ������
y_ft=fft(y,nfft);%��y�źŽ���DFT���õ�Ƶ�ʵķ�ֵ�ֲ�
y_p=y_ft.*conj(y_ft)/nfft;%conj()��������y�����Ĺ������ʵ���Ĺ������������
y_f=fs*(0:nfft/2-1)/nfft;%T�任���Ӧ��Ƶ�ʵ�����
% y_p=y_ft.*conj(y_ft)/nfft;%conj()��������y�����Ĺ������ʵ���Ĺ������������
if style==1
    if nargin==3
        cc=2*abs(y_ft(1:nfft/2))/length(y);
        %ylabel('��ֵ');xlabel('Ƶ��');title('�źŷ�ֵ��');
        %plot(y_f,abs(y_ft(1:nfft/2)));%��̳�ϻ�FFT�ķ���
    else
        f1=varargin{1};
        fn=varargin{2};
        ni=round(f1 * nfft/fs+1);
        na=round(fn * nfft/fs+1);
        hold on
        plot(y_f(ni:na),abs(y_ft(ni:na)*2/nfft),'k');
    end
elseif style==2
            plot(y_f,y_p(1:nfft/2),'k');
            %ylabel('�������ܶ�');xlabel('Ƶ��');title('�źŹ�����');
    else
        subplot(211);plot(y_f,2*abs(y_ft(1:nfft/2))/length(y),'k');
        ylabel('��ֵ');xlabel('Ƶ��');title('�źŷ�ֵ��');
        subplot(212);plot(y_f,y_p(1:nfft/2),'k');
        ylabel('�������ܶ�');xlabel('Ƶ��');title('�źŹ�����');
end
end

