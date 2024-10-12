clc;
clear;
close all;

N = 1000;
M = 500;
m = 1e-2;
seed =777;
dimM = 1000;
dimN = 5000;
%density=0.05;
%r=1;
[F, PAR] = test_fn_unconstr_01(N, M, m, seed, dimM, dimN);
%[F, PAR] = test_fn_unconstr_box(N, M, m, seed, dimM, dimN,r);
%[F, PAR] = test_fn_unconstr_sdp(N, M, m, seed, dimM, dimN, density);
Lipschitz=max(sort(eig(F.hess())))
%Lipschitz=2*10^4;

para.tol=(PAR.norm_fn(F.grad(PAR.x0))+1)*1e-13;
para.n=dimN;
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
tic
[x, its, dk, ek, fk, cnt] = func_Rada_FISTA(p,q,r, para, ProxJ,GradF, ObjPhi);
toc

function [x, its, dk, ek, fk, cnt] = func_Rada_FISTA(p,q,r, para, ProxJ,GradF, ObjPhi)
% Rada FISTA

n = para.n;
% J = para.J;
mu = para.mu;
gamma = para.gamma;
tol = para.tol;
maxits = para.maxits  + 1;

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
    FuncValue=ObjPhi(x)
    if NormCompute<=tol; break; end
    
    its = its + 1
    
end
fprintf('\n');

% r
% xi

dk = dk(1:its-1);
ek = ek(1:its-1);
fk = fk(1:its-1);

end

% EoF