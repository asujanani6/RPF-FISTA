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
% filename='GreedyFISTA_vec_simplex_7-12.xlsx';

%%Updated
N = [1000,1000,1000,1000,1000,1000,1000,100,500,5000,1000,1000];
M = [100,500,1000,5000,10000,1e6,1000,5e3,5e4,1e6,1e2,1e4];
m = [1e-4,1e-2,1,1e-2,1e-3,1,10,1e1,1e-1,1,10,1];
seed =[777,777,777,777,777,777,777,777,777,777,777,777];
Problems=[1,2,3,4,5,6,7,8,9,10,11,12];
dimM = [1000,1000,1000,1000,1000,1000,2000,2000,2000,2000,2000,2000];
dimN = [5000,5000,5000,5000,5000,5000,10000,10000,10000,10000,10000,10000];
time_limit=7200;
filename='GreedyFISTA_vec_simplex_6_108.xlsx';

for k=6
    Problem=Problems(k)
    [F, PAR] = test_fn_unconstr_01(N(k), M(k), m(k), seed(k), dimM(k), dimN(k));
    %[F, PAR] = test_fn_unconstr_box(N, M, m, seed, dimM, dimN,r);
    Lipschitz=max(sort(eig(F.hess())))

    para.initialnorm=norm(F.grad(PAR.x0))+1;
    para.tol=para.initialnorm*1e-8;
    para.n=dimN;
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