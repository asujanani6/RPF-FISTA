clc;
clear;
close all;

for i=1:60
    seed(i)=777;
end
for i=1:20
    for j=1:3
        if i==1
            name(i+j-1)="aa03";
        elseif i==2
            name(i+j+1)="aa3";
        elseif i==3
            name(i+j+3)="aa4";
        elseif i==4
            name(i+j+5)="air02";
        elseif i==5
            name(i+j+7)="air03";
        elseif i==6
            name(i+j+9)="air04";
        elseif i==7
            name(i+j+11)="air05";
        elseif i==8
            name(i+j+13)="air06";
        elseif i==9
            name(i+j+15)="gen";
        elseif i==10
            name(i+j+17)="gen1";
        elseif i==11
            name(i+j+19)="us04";
        elseif i==12
            name(i+j+21)="rosen1";
        elseif i==13
            name(i+j+23)="rosen10";
        elseif i==14
            name(i+j+25)="crew1";
        elseif i==15
            name(i+j+27)="cr42";
        elseif i==16
            name(i+j+29)="kl02";
        elseif i==17
            name(i+j+31)="t0331-4l";
        elseif i==18
            name(i+j+33)="rail507";
        elseif i==19
            name(i+j+35)="rail516";
        elseif i==20
            name(i+j+37)="rail582";
        end

    end

end

for i=1:60
    if mod(i,3)==1
        r(i)=1;
    elseif mod(i,3)==2
        r(i)=5;
    elseif mod(i,3)==0
        r(i)=10;
    end

end

for i=1:60
    Problems(i)=i;
end

time_limit=7200;

filename='GreedyFISTA_LS_Real_1013.xlsx';

for k=1:60
    Problem=Problems(k)
    
    [F, PAR] = test_fn_l1ball_leastsquaresRealData(seed(k),name(k),r(k))

    Lipschitz=PAR.M1

    para.initialnorm=norm(F.grad(PAR.x0))+1;
    para.tol=para.initialnorm*1e-13;
    para.n=PAR.n;
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
    writematrix(Iterations,filename,'Sheet',1,'Range','A75');
    writematrix(ResidualNorm,filename,'Sheet',1,'Range','A150');
    writematrix(FunctionValue,filename,'Sheet',1,'Range','A230');
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