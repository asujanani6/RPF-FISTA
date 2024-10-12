clc;
clear;
close all;

seed =[200,200,200,100,100,100,1000,1000,1000,500,500,500];
nInput = [50000,50000,50000,250000,250000,250000,500000,500000,500000,1000000,1000000,1000000];
mInput = [500,500,500,1000,1000,1000,300,300,300,100,100,100];
rInput = [0.5,1,2,0.5,1,2,0.5,1,2,0.5,1,2];
Problems=[1,2,3,4,5,6,7,8,9,10,11,12];
time_limit=7200;


filename='RADAFISTA_vec_logL1_1_12_108.xlsx';
for k=1:12
    Problem=Problems(k)
    %[F, PAR]= test_fn_unconstr_hyper(N(k), M(k), m(k),seed(k), dimM(k),dimN(k),avec,bscal,lower,upper);
    [F, PAR] = test_fn_l1ball_log_regression(seed(k),mInput(k),nInput(k),rInput(k));
    
    Lipschitz=PAR.M1
    
   

    para.initialnorm=norm(F.grad(PAR.x0))+1;
    para.tol=para.initialnorm*1e-8;
    para.n=nInput(k);
    para.gamma=1/Lipschitz;
    para.maxits=1e6;
    para.mu=0;
    para.x0=PAR.x0;
    p=0.5;
    q=0.5;
    r=4;
    ProxJ=F.prox;
    GradF=F.grad;
    ObjPhi=F.fs;
    [x, its, dk, ek, fk, cnt, NormCompute, FuncValue, time] = func_Rada_FISTA(p,q,r, para, ProxJ, GradF, ObjPhi, time_limit);
    TimeFinal(k,1)=time;
    Iterations(k,1)=its;
    ResidualNorm(k,1)=NormCompute;
    FunctionValue(k,1)=FuncValue;
    RestartsNum(k,1)=cnt;

    writematrix(TimeFinal,filename,'Sheet',1,'Range','A1');
    writematrix(Iterations,filename,'Sheet',1,'Range','A15');
    writematrix(ResidualNorm,filename,'Sheet',1,'Range','A30');
    writematrix(FunctionValue,filename,'Sheet',1,'Range','A45');
    writematrix(RestartsNum,filename,'Sheet',1,'Range','A60');
end


function [x, its, dk, ek, fk, cnt, NormCompute, FuncValue, time] = func_Rada_FISTA(p,q,r, para, ProxJ, GradF, ObjPhi, time_limit)
% Rada FISTA
para.initialnorm
n = para.n;
% J = para.J;
mu = para.mu;
gamma = para.gamma;
tol = para.tol;
maxits = para.maxits  + 1;
NormComputeOld=Inf;
tau = mu*gamma;

% Forward--Backward Operator
FBO = @(y) ProxJ(y, tau);
%FBO = @(y) ProxJ(y-gamma*GradF(y), tau);
%% FBS iteration
x0 = para.x0;
x = x0;
y = x0;

dk = zeros(maxits, 1);
ek = zeros(maxits, 1);
fk = zeros(maxits, 1);

t = 1;
tor = 0;
cnt = 0;
flag = 1;

its = 1;
tic
while(its<maxits)
    
    
    x_old = x;
    y_old = y;
    Grady=GradF(y);
    x=FBO(y-gamma*Grady);
    %x = FBO(y);
    
    t_old = t;
    t = (p + sqrt(q+r*t_old^2)) /2;
    a = min(1, (t_old-1) /t);
    
    y = x + a*(x-x_old);
    
    %%% update r_k
    norm_ = norm(y_old(:)-x(:)) * norm(x(:)-x_old(:));
    vk = (y_old(:)-x(:))'*(x(:)-x_old(:));
    if vk >= tor* norm_
        
        cnt = cnt + 1;
        
        if cnt>=4 % increase the value here if the condition number is big
            
            if flag
                a_half = (4+1*a) /5;
                xi = a_half^(1/30);
                
                flag = 0;
            end
            
            r = r * xi;
            
            if r<3.99
                t_lim = ( 2*p + sqrt( r*p^2 + (4-r)*q ) ) / (4 - r);
                t = max(2 * t_lim, t);
            end
            
        else
            % t = 1;
        end
        
        y = x;
        
    end
    
    %%%%%%% stop?
    res = norm(x_old-x, 'fro');
    
    
    ek(its) = res;
    %NormCompute=norm((1/gamma)*(y_old-x)-GradF(y_old)+GradF(x), 'fro')
    NormCompute=norm((1/gamma)*(y_old-x)-Grady+GradF(x), 'fro')

    NormMin=min(NormCompute,NormComputeOld);
    NormComputeOld=NormCompute;

    FuncValue=ObjPhi(x)
    if toc>=time_limit
        time=toc;
        NormCompute=(NormMin/para.initialnorm);
        break
    end

    if NormCompute<=tol; break; end
    
    its = its + 1
    
end
time=toc;
fprintf('\n');

% r
% xi

dk = dk(1:its-1);
ek = ek(1:its-1);
fk = fk(1:its-1);

end

% EoF