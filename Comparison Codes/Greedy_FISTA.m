clc;
clear;
close all;

% N = 1000;
% M = 1000;
% m = 100;
% seed =777;
% dimM = 500;
% dimN = 1000;
% %r=1;
% avec=ones(dimN,1);
% %avec(end)=-1;
% avec(end-10:end)=-1;
% bscal=-10;
% lower=-1;
% upper=1;
seed=1000;
n = 100;
m = 200;
r=0.5;

%[F, PAR] = test_fn_unconstr_01(N, M, m, seed, dimM, dimN);
%[F, PAR]= test_fn_unconstr_hyper(N, M, m,seed,dimM,dimN,avec,bscal,lower,upper);
%[F, PAR] = test_fn_unconstr_box(N, M, m, seed, dimM, dimN,r);
[F, PAR] = test_fn_l1ball_log_regression(seed,m,n,r)
%Lipschitz=max(sort(eig(F.hess())))
Lipschitz=PAR.M1

para.tol=(norm(F.grad(PAR.x0))+1)*1e-8;
%para.n=dimN;
para.n=n;
para.gamma=1/Lipschitz;
para.c_gamma = 1.3;
para.maxits=1e6;
para.mu=0;
para.x0=PAR.x0;
para.a=@(k) 1.0;
ProxJ=F.prox;
GradF=F.grad;
ObjPhi=F.fs;
tic
[x, its, dk, ek, fk] = func_Greedy_FISTA(para, ProxJ,GradF, ObjPhi);
toc

function [x, its, dk, ek, fk] = func_Greedy_FISTA(para, ProxJ,GradF, ObjPhi)
% Greedy FISTA

n = para.n;
% J = para.J;
mu = para.mu;
gamma0 = para.gamma;
gamma = para.c_gamma* gamma0;
tol = para.tol;
maxits = para.maxits  + 1;

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
    FuncValue=ObjPhi(x)
    if  NormCompute<=tol; break; end
    
    %%% safeguard
     if res>S*ek(1); gamma = max(gamma0, gamma*xi); end % x = x_old; y = x_old; end
    
    its = its + 1
    
end
fprintf('\n');

% if verbose; fprintf('\n'); disp(gamma/para.gamma); fprintf('\n'); end

dk = dk(1:its-1);
ek = ek(1:its-1);
fk = fk(1:its-1);
end
% EoF