clc;
clear;
close all;

% N = [1000,1000,1000,1000,5000,1000,1000,100,100,500,1000,1000];
% M = [100,500,1000,1000,1e10,1e11,1000,1e4,1e7,1e9,1e2,1e4];
% m = [1e-4,1e-2,1,100,1e3,1e3,10,100,1e4,1e5,10,1];
% seed =[777,777,777,777,777,777,777,777,777,777,777,777];
% Problems=[1,2,3,4,5,6,7,8,9,10,11,12];
% dimM = [1000,1000,1000,1000,1000,1000,2000,2000,2000,2000,2000,2000];
% dimN = [5000,5000,5000,5000,5000,5000,10000,10000,10000,10000,10000,10000];
% %r=1;
% time_limit=7200;
%filename='BeckFISTA_vec_simplex_11_12.xlsx';

%%Updated
N = [1000,1000,1000,1000,1000,1000,1000,100,500,5000,1000,1000];
M = [100,500,1000,5000,10000,1e6,1000,5e3,5e4,1e6,1e2,1e4];
m = [1e-4,1e-2,1,1e-2,1e-3,1,10,1e1,1e-1,1,10,1];
seed =[777,777,777,777,777,777,777,777,777,777,777,777];
Problems=[1,2,3,4,5,6,7,8,9,10,11,12];
dimM = [1000,1000,1000,1000,1000,1000,2000,2000,2000,2000,2000,2000];
dimN = [5000,5000,5000,5000,5000,5000,10000,10000,10000,10000,10000,10000];
time_limit=7200;
filename='BeckFISTA_vec_simplex_1_12_1013.xlsx';





for k=1:12
    Problem=Problems(k)
    [F, PAR] = test_fn_unconstr_01(N(k), M(k), m(k), seed(k), dimM(k), dimN(k));

    PAR.initialnorm=norm(F.grad(PAR.x0))+1;
    PAR.L0=10;
    PAR.mu=0;
    PAR.chi=0.001;
    PAR.beta=1.25;
    PAR.last=1;
    tol=(norm(F.grad(PAR.x0))+1)*1e-13

    [y,L,FuncValue,NormCompute,iter,time]=FISTA(F,PAR,tol,time_limit);
    
    TimeFinal(k,1)=time;
    Iterations(k,1)=iter;
    ResidualNorm(k,1)=NormCompute;
    FunctionValue(k,1)=FuncValue;

    writematrix(TimeFinal,filename,'Sheet',1,'Range','A1');
    writematrix(Iterations,filename,'Sheet',1,'Range','A15');
    writematrix(ResidualNorm,filename,'Sheet',1,'Range','A30');
    writematrix(FunctionValue,filename,'Sheet',1,'Range','A45');
end










function [y,L,FuncValue,NormCompute,iter,time]=FISTA(F,PAR,tol,time_limit)
iter=0;
x=PAR.x0;
y=PAR.x0;
A=0;
tau=1;
L=PAR.L0;
sumAL=0;
fistaFailure=0;
NormComputeOld=Inf;

tic
for j=1:1e6
    a=(tau+sqrt(tau^2+4*tau*A*L))/(2*L);
    xtilde=((A*y)+(a*x))/(A+a);
    storegradient=F.grad(xtilde);
    y=F.prox(xtilde-(1/L)*storegradient);
    iter=iter+1

    if F.fs(xtilde)+PAR.prod_fn(y-xtilde,storegradient)+((1-PAR.chi)*L*PAR.norm_fn(y-xtilde)^2)/(4)+10^-6<F.fs(y)
        success=0;
        L=PAR.beta*L
    else
        success=1;
    end

    while success==0
        a=(tau+sqrt(tau^2+4*tau*A*L))/(2*L);
        xtilde=((A*y)+(a*x))/(A+a);
        storegradient=F.grad(xtilde);
        y=F.prox(xtilde-(1/L)*storegradient);
        iter=iter+1;
        if F.fs(xtilde)+PAR.prod_fn(y-xtilde,storegradient)+((1-PAR.chi)*L*PAR.norm_fn(y-xtilde)^2)/(4)+10^-6<F.fs(y)
            success=0;
            L=PAR.beta*L
        else
            success=1;
        end
    end

    s=L*(xtilde-y);
    u=F.grad(y)-storegradient+s;
    storegradient=[];
    A=A+a;
    oldtau=tau;
    tau=oldtau+(0.5*PAR.mu*a);
    x=(1/tau)*((0.5*PAR.mu*a*y)+(oldtau*x)-(a*s));


   
    NormCompute=PAR.norm_fn(u)

    NormMin=min(NormCompute,NormComputeOld);
    NormComputeOld=NormCompute;

    FuncValue=F.fs(y)

    if toc>=time_limit
        time=toc;
        NormCompute=(NormMin/PAR.initialnorm);
        break
    end
    

    if NormCompute<=tol
        break
    end




end
time=toc;
end