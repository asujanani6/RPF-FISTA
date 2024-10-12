% Solve a multivariate nonconvex quadratic programming problem constrained to the unit simplex

% The function of interest is
%
%  f(x) :=  -xi / 2 * ||D * B * x|| ^ 2 + tau / 2 * ||A * x - b|| ^ 2
%
% with curvature pair (m, M).

% Use a problem instance generator to create the oracle and hyperparameters.
clc;
clear;
close all;

% N = 5000;
% M = 1e6;
% m = 1e0;
% seed =777;
% dimM = 2000;
% dimN = 10000;

seed=777
name="aa03"
r=1
[F, PAR] = test_fn_l1ball_leastsquaresRealData(seed,name,r)


% seed=200;
% n = 50000;
% m = 500;
% r=0.5;

% seed=500;
% n = 30000;
% m = 300;
% r=0.5;

% avec=ones(dimN,1);
% avec(end)=-1;
% %avec(end-30:end)=-1;
% bscal=0;
% lower=-1;
% upper=1;
%r=5;
% [F, PAR] = test_fn_unconstr_01(N, M, m, seed, dimM, dimN);
% [F, PAR] = test_fn_l1ball_log_regression(seed,m,n,r)
%[F, PAR] = test_fn_l1ball_leastsquares(seed,m,n,r)
%[F, PAR]= test_fn_unconstr_hyper(N, M, m,seed,dimM,dimN,avec,bscal,lower,upper);
%[F, PAR] = test_fn_unconstr_box(N, M, m, seed, dimM, dimN,r);
%eigenvalues=sort(eig(F.hess()));

% x0=PAR.x0;
% fun1=F.fs(x0);
% fun2=F.fn(x0);
% fun3=F.grad(x0);
% fun4=F.prox(x0);
% hess=F.hess(PAR.x0);
% Eigen=sort(eig(hess));

%Out=Testing(2,3)
PAR.L0=10;
PAR.mu=0;
PAR.chi=0.001;
PAR.beta=1.25;
PAR.last=1;
tol=(norm(F.grad(PAR.x0))+1)*1e-13

fistaFailure=1;
totalIter=0;
Restarts=-1;
tic
while fistaFailure==1
Restarts=Restarts+1;
[y,L,fistaFailure,iter]=FISTACand(F,PAR,tol);
totalIter=totalIter+iter
if fistaFailure==0
    PAR.x0=y;
    PAR.L0=L;
    PAR.mu=PAR.mu;
    break
end
PAR.x0=y;
PAR.mu=0.1*PAR.mu
PAR.L0=L;
end
toc


function [y,L,fistaFailure,iter]=FISTACand(F,PAR,tol)
iter=0;
x=PAR.x0;
y=PAR.x0;
A=0;
tau=1;
L=PAR.L0;
sumAL=0;
fistaFailure=0;

for j=1:1e6
    a=(tau+sqrt(tau^2+4*tau*A*L))/(2*L);
    xtilde=((A*y)+(a*x))/(A+a);
    storegradient=F.grad(xtilde);
    yold=y;
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
   
   if F.fs(y)-1e-6>F.fs(yold) && j>=2
            fistaFailure=1;
            y=yold;
            break
   end
    s=L*(xtilde-y);
    u=F.grad(y)-storegradient+s;
    storegradient=[];
    A=A+a;
    oldtau=tau;
    tau=oldtau+(0.5*PAR.mu*a);
    x=(1/tau)*((0.5*PAR.mu*a*y)+(oldtau*x)-(a*s));


   
    Normu=PAR.norm_fn(u)
    FuncValue=F.fs(y)

    if PAR.norm_fn(u)<=tol
        break
    end




end

end