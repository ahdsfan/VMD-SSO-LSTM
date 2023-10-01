function [u, u_hat, omega] = VMD(signal, alpha, tau, K, DC, init, tol)
% Variational Mode Decomposition——VMD
% Authors: Konstantin Dragomiretskiy and Dominique Zosso
% zosso@math.ucla.edu --- http://www.math.ucla.edu/~zosso
% Initial release 2013-12-12 (c) 2013
%
% Input and Parameters:     输入与参数
% ---------------------
% signal  - the time domain signal (1D) to be decomposed                待分解的时域信号
% alpha   - the balancing parameter of the data-fidelity constraint     a-数据保真度约束的平衡参数
% tau     - time-step of the dual ascent ( pick 0 for noise-slack )     T-双上升的时间步长(噪声松弛取0)
% K       - the number of modes to be recovered
% DC      - true if the first mode is put and kept at DC (0-freq)       如果第一种模式被放置并保持在DC(0频率)，则为真
% init    - 0 = all omegas start at 0                                   所有从0开始
%                    1 = all omegas start uniformly distributed         1 =所有都是均匀分布的
%                    2 = all omegas initialized randomly                2 =随机初始化的所有
% tol     - tolerance of convergence criterion; typically around 1e-6   收敛公差准则;通常大约e-6
%
% Output:                    输出
% -------
% u       - the collection of decomposed modes                             采集分解模态
% u_hat   - spectra of the modes                                                   模态的范围
% omega   - estimated mode center-frequencies                           估算模态的中心频率
%
% When using this code, please do cite our paper:
% -----------------------------------------------
% K. Dragomiretskiy, D. Zosso, Variational Mode Decomposition, IEEE Trans.
% on Signal Processing (in press)
% please check here for update reference: 
%          http://dx.doi.org/10.1109/TSP.2013.2288675

%% Preparations         准备

% Period and sampling frequency of input signal           输入信号的周期和采样频率
save_T = length(signal);                       %采样周期/采样点数
fs = 1/save_T;                                      %采样频率

% extend the signal by mirroring                          通过镜像扩展信号
T = save_T;                                           
f_mirror(1:T/2) = signal(T/2:-1:1);        %首项T/2,尾箱1，公差为-1的·等差数列。倒这
f_mirror(T/2+1:3*T/2) = signal;
f_mirror(3*T/2+1:2*T) = signal(T:-1:T/2+1);
%% 镜像函数处理
f = f_mirror;                                             %周期扩展为原来的2倍

% Time Domain 0 to T (of mirrored signal)                 时域0到T（镜像信号）
T = length(f);                                            %Tm的周期为2T/采样点数
t = (1:T)/T;                                              %时间间隔/频率  镜像信号的采样点数

% Spectral Domain discretization                          频域离散化
freqs = t-0.5-1/T;                                       %频率范围[-0.5 0.5-1/T]

% Maximum number of iterations (if not converged yet, then it won't anyway)
%最大迭代次数(如果还没有收敛，那么无论如何它都不会收敛)
N = 500;

% For future generalizations: individual alpha for each mode 对于未来的概括:每个模态的alpha值
Alpha = alpha*ones(1,K);                    

% Construct and center f_hat                构造并且居中f_hat   傅里叶变换，求其频谱，并在频域进行泛函的更迭
f_hat = fftshift((fft(f)));                           
f_hat_plus = f_hat;                                        
f_hat_plus(1:T/2) = 0;                                     %清除负频部分只保留正频部分

% matrix keeping track of every iterant // could be discarded for mem  
%矩阵跟踪每一个迭代器//可以为mem丢弃            初始化模态  
u_hat_plus = zeros(N, length(freqs), K);                   %建立一个N*length(freqs)*K的三维0数组               

% Initialization of omega_k                   初始化omega_k
omega_plus = zeros(N, K);                                  %设置 其最大宽度N 及 模态个数
switch init
    case 1
        for i = 1:K
            omega_plus(1,i) = (0.5/K)*(i-1);
        end
    case 2
        omega_plus(1,:) = sort(exp(log(fs) + (log(0.5)-log(fs))*rand(1,K)));
    otherwise
        omega_plus(1,:) = 0;
end

% if DC mode imposed, set its omega to 0         采用DC模式，omega设为0
if DC
    omega_plus(1,1) = 0;
end

% start with empty dual variables                从空的对偶变量开始         
lambda_hat = zeros(N, length(freqs));

% other inits
uDiff = tol+eps;                    % update step     步骤更新
n = 1;                              % loop counter     设置循环计数器为1
sum_uk = 0;                         % accumulator  累加器



% ----------- Main loop for iterative updates           对于迭代更新的主循环
          %Uk(w)       Wk(w)  更新
while ( uDiff > tol &&  n < N ) % not converged and below iterations limit   不收敛且小于迭代极限  N=500
    
    % update first mode accumulator                   第一个模式
    k = 1;
    sum_uk = u_hat_plus(n,:,K) + sum_uk - u_hat_plus(n,:,1);
    
    % update spectrum of first mode through Wiener filter of residuals    通过残差维纳滤波更新第一模态频谱
    u_hat_plus(n+1,:,k) = (f_hat_plus - sum_uk - lambda_hat(n,:)/2)./(1+Alpha(1,k)*(freqs - omega_plus(n,k)).^2);
    
    % update first omega if not held at 0            如果未保持在0，则更新第一个omega
    if ~DC
        omega_plus(n+1,k) = (freqs(T/2+1:T)*(abs(u_hat_plus(n+1, T/2+1:T, k)).^2)')/sum(abs(u_hat_plus(n+1,T/2+1:T,k)).^2);
    end
    
    % update of any other mode                       更新其他模式2-K模式
    for k=2:K
        
        % accumulator
        sum_uk = u_hat_plus(n+1,:,k-1) + sum_uk - u_hat_plus(n,:,k);
        
        % mode spectrum                              模态频谱
        u_hat_plus(n+1,:,k) = (f_hat_plus - sum_uk - lambda_hat(n,:)/2)./(1+Alpha(1,k)*(freqs - omega_plus(n,k)).^2);
        
        % center frequencies                         中心频率          
        omega_plus(n+1,k) = (freqs(T/2+1:T)*(abs(u_hat_plus(n+1, T/2+1:T, k)).^2)')/sum(abs(u_hat_plus(n+1,T/2+1:T,k)).^2);
        
    end
    
    % Dual ascent                              双提升
    lambda_hat(n+1,:) = lambda_hat(n,:) + tau*(sum(u_hat_plus(n+1,:,:),3) - f_hat_plus);
    
    % loop counter                             累加器
    n = n+1;
    
    % converged yet?                         是否满足迭代停止条件                   
    uDiff = eps;
    for i=1:K
        uDiff = uDiff + 1/T*(u_hat_plus(n,:,i)-u_hat_plus(n-1,:,i))*conj((u_hat_plus(n,:,i)-u_hat_plus(n-1,:,i)))';
    end
    uDiff = abs(uDiff);
    
end




%% Postprocessing and cleanup          后续清理，即输出时域模态及频域模态

% discard empty space if converged early      如果早收敛，则丢弃空空间
N = min(N,n);                                                   %程序实际迭代次数n         
omega = omega_plus(1:N,:);                             %输出omega，实际长度为程序循环次数，个数为K      频率范围是固定的

% Signal reconstruction  重建镜像的频域模态信号，此时重建的是1：T内的频域模态信号，即模态的解析信号即有是不也有虚部
u_hat = zeros(T, K);                                                       %重建信号为2T*K的矩阵
u_hat((T/2+1):T,:) = squeeze(u_hat_plus(N,(T/2+1):T,:));                   %重建采样点T/2+1:T的信号      对模态进行降维，删除长度为1的维度 
u_hat((T/2+1):-1:2,:) = squeeze(conj(u_hat_plus(N,(T/2+1):T,:)));          %重建采样点2:T/2+1的信号
u_hat(1,:) = conj(u_hat(end,:));                                           %重建采样点1的信号            取共轭复数
%重建镜像的时域模态信号
u = zeros(K,length(t));                                                   
for k = 1:K
    u(k,:)=real(ifft(ifftshift(u_hat(:,k))));                              %保留频域模态信号进行反变化并取其实数部分
end

%%  remove mirror part                         重建实际信号的时域模态，清楚镜像部分
u = u(:,T/4+1:3*T/4);                            %这里的T还是Tm，这里只取了Tm周期内的T/4+1:3*T/4

% recompute spectrum                         重建实际信号的频域模态
clear u_hat;
for k = 1:K
    u_hat(:,k)=fftshift(fft(u(k,:)))';            
end

end
