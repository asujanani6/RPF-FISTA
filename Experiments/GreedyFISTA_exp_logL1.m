clc;
clear;
close all;

seed =[200,200,200,100,100,100,1000,1000,1000,500,500,500];
n = [50000,50000,50000,250000,250000,250000,500000,500000,500000,1000000,1000000,1000000];
m = [500,500,500,1000,1000,1000,300,300,300,100,100,100];
r = [0.5,1,2,0.5,1,2,0.5,1,2,0.5,1,2];
Problems=[1,2,3,4,5,6,7,8,9,10,11,12];
time_limit=7200;
%filename='GreedyFISTA_vec_boxH_1-12.xlsx';
filename='GreedyFISTA_vec_logL1_9_12_108.xlsx';
for k=9:12
    Problem=Problems(k)
    %[F, PAR]= test_fn_unconstr_hyper(N(k), M(k), m(k),seed(k), dimM(k),dimN(k),avec,bscal,lower,upper);
    [F, PAR] = test_fn_l1ball_log_regression(seed(k),m(k),n(k),r(k));
    
    Lipschitz=PAR.M1

    para.initialnorm=norm(F.grad(PAR.x0))+1;
    para.tol=para.initialnorm*1e-8;
    para.n=n(k);
    para.gamma=1/Lipschitz;
    para.c_gamma = 1.3;
    para.maxits=1e6;
    para.mu=0;
    para.x0=PAR.x0;
    para.a=@(k) 1.0;
    ProxJ=F.prox;
    GradF=F.grad;
    ObjPhi=F.fs;
    [x, its, dk, ek, fk, NormCompute, FuncValue, time] = func_Greedy_FISTA(para, ProxJ,GradF, ObjPhi, time_limit);
    Time(k,1)=time;
    Iterations(k,1)=its;
    ResidualNorm(k,1)=NormCompute;
    FunctionValue(k,1)=FuncValue;

    writematrix(Time,filename,'Sheet',1,'Range','A1');
    writematrix(Iterations,filename,'Sheet',1,'Range','A15');
    writematrix(ResidualNorm,filename,'Sheet',1,'Range','A30');
    writematrix(FunctionValue,filename,'Sheet',1,'Range','A45');
end

function [x, its, dk, ek, fk, NormCompute, FuncValue, time] = func_Greedy_FISTA(para, ProxJ,GradF, ObjPhi, time_limit)
% Greedy FISTA
para.initialnorm
n = para.n;
% J = para.J;
mu = para.mu;
gamma0 = para.gamma;
gamma = para.c_gamma* gamma0;
tol = para.tol;
maxits = para.maxits  + 1;
NormComputeOld=Inf;
% tau = mu*gamma;

% Forward--Backward Operator
FBO = @(y, gamma) ProxJ(y, mu*gamma);
%% FBS iteration
x0 = para.x0;
x = x0;
y = x0;

dk = zeros(maxits, 1);
ek = zeros(maxits, 1);
fk = zeros(maxits, 1);

a = para.a;

tor = 0;
S = 1;
xi = 0.96;
% first = 1;
% e0 = 1e5;

its = 1;
tic
while(its<maxits)


    x_old = x;
    y_old = y;

    Grady=GradF(y);
    x = FBO(y-gamma*Grady, gamma);

    y = x + a(its)*(x-x_old);

    %%% gradient criteria
    norm_ = norm(y_old(:)-x(:)) * norm(x(:)-x_old(:));
    vk = (y_old(:)-x(:))'*(x(:)-x_old(:));
    if vk >= tor* norm_
        y = x;
        % if first; e0 = max(ek(1:its)); first = 0; end
    end

    %%%%%%% stop?
    res = norm(x_old-x, 'fro');


    ek(its) = res;

    NormCompute=norm((1/gamma)*(y_old-x)-Grady+GradF(x), 'fro')
    
    NormMin=min(NormCompute,NormComputeOld);
    NormComputeOld=NormCompute;

    FuncValue=ObjPhi(x)
    if toc>=time_limit
        time=toc;
        NormCompute=(NormMin/para.initialnorm);
        break
    end
    if  NormCompute<=tol; break; end

    %%% safeguard
    if res>S*ek(1); gamma = max(gamma0, gamma*xi); end % x = x_old; y = x_old; end

    its = its + 1

end
time=toc;
fprintf('\n');

% if verbose; fprintf('\n'); disp(gamma/para.gamma); fprintf('\n'); end

dk = dk(1:its-1);
ek = ek(1:its-1);
fk = fk(1:its-1);
end
% EoF