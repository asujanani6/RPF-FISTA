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
filename='CandesFISTA_LS_Real47_60_1013.xlsx';


for k=47:60
    Problem=Problems(k)
    [F, PAR] = test_fn_l1ball_leastsquaresRealData(seed(k),name(k),r(k))

    PAR.initialnorm=norm(F.grad(PAR.x0))+1;
    PAR.L0=10;
    PAR.mu=0;
    PAR.chi=0.001;
    PAR.beta=1.25;
    PAR.last=1;
    tol=(norm(F.grad(PAR.x0))+1)*1e-13

    [y,L,FuncValue,NormComputeMin,totalIter,totalTime,Restarts]=RestartedFISTA(F,PAR,tol,time_limit);


    TimeFinal(k,1)=totalTime;
    Iterations(k,1)=totalIter;
    ResidualNorm(k,1)=NormComputeMin;
    FunctionValue(k,1)=FuncValue;
    NumberRestarts(k,1)=Restarts;

    writematrix(TimeFinal,filename,'Sheet',1,'Range','A1');
    writematrix(Iterations,filename,'Sheet',1,'Range','A80');
    writematrix(ResidualNorm,filename,'Sheet',1,'Range','A160');
    writematrix(FunctionValue,filename,'Sheet',1,'Range','A240');
    writematrix(NumberRestarts,filename,'Sheet',1,'Range','A320');

end



function [y,L,FuncValue,NormComputeMin,totalIter,totalTime,Restarts]=RestartedFISTA(F,PAR,tol,time_limit)

fistaFailure=1;
totalIter=0;
totalTime=0;
Restarts=-1;
NormComputeOld=Inf;

while fistaFailure==1
    Restarts=Restarts+1;
    [y,L,FuncValue,NormCompute,fistaFailure,iter,totalTime,FLAGTime]=FISTACandes(F,PAR,tol,time_limit,totalTime);

    NormComputeMin=min(NormComputeOld,NormCompute);
    NormComputeOld=NormCompute;

    totalIter=totalIter+iter
    if fistaFailure==0 || FLAGTime==1
        break
    end
    PAR.x0=y;
    PAR.mu=0.1*PAR.mu
    PAR.L0=L;
end

end




function [y,L,FuncValue,NormCompute,fistaFailure,iter,time,FLAGTime]=FISTACandes(F,PAR,tol,time_limit,totalTime)
iter=0;
FLAGTime=0;
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
    yold=y;
    FuncValueOld=F.fs(yold);

    y=F.prox(xtilde-(1/L)*storegradient);
    iter=iter+1
    FuncValue=F.fs(y)
    NormDiff=PAR.norm_fn(y-xtilde)^2;

    if F.fs(xtilde)+PAR.prod_fn(y-xtilde,storegradient)+((1-PAR.chi)*L*NormDiff)/(4)+10^-6<FuncValue
        success=0;
        L=PAR.beta*L;
    else
        success=1;
    end
    while success==0
        a=(tau+sqrt(tau^2+4*tau*A*L))/(2*L);
        xtilde=((A*y)+(a*x))/(A+a);
        storegradient=F.grad(xtilde);
        y=F.prox(xtilde-(1/L)*storegradient);
        iter=iter+1;
        FuncValue=F.fs(y);
        NormDiff=PAR.norm_fn(y-xtilde)^2;
        if F.fs(xtilde)+PAR.prod_fn(y-xtilde,storegradient)+((1-PAR.chi)*L*NormDiff)/(4)+10^-6<FuncValue
            success=0;
            L=PAR.beta*L;
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

    if FuncValue-1e-6>FuncValueOld && j>=2
        fistaFailure=1;
        y=yold;
        NormCompute=(NormMin/PAR.initialnorm);
        break
    end

    if totalTime+toc>=time_limit
        NormCompute=(NormMin/PAR.initialnorm);
        FLAGTime=1;
        break
    end

    if NormCompute<=tol
        NormCompute=(NormMin/PAR.initialnorm);
        break
    end




end
time=toc+totalTime;
end
