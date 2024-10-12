% SPDX-License-Identifier: MIT
% Copyright © 2021 Weiwei "William" Kong

function [F, params] = test_fn_l1ball_log_regression(seed,m,n,r)
% Generator of a test suite of unconstrained logistic regression.
%
%
%   seed (int): The number used to seed MATLAB's random number generator.
%
% Returns:
%
%   A pair consisting of an Oracle and a struct. The oracle is first-order oracle underyling the optimization problem and the
%   struct contains the relevant hyperparameters of the problem.
%

% Initialize
rng(seed);

A = rand(m, n);
b = randi([0, 1], m, 1) * 2 - 1;
ba = A' * diag(b)';

x0 = rand(n, 1) * 2 - 1;

% Set the topology (Euclidean)
prod_fn = @(a,b) sum(dot(a, b));
norm_fn = @(a) norm(a, 'fro');

% Output params
params.prod_fn = prod_fn;
params.norm_fn = norm_fn;
params.M1=norm(ba)^2/4;
params.m = -1;

% params.M2 = norm(A'*b)^2/4;
% params.M3=norm(A, 2)^2/4;

% Create the Oracle object
f_s = @(x) sum(log_exp(ba, x));
f_n = @(x) 0;
grad_f_s = @(x) -ba*log_func(ba,x);
prox_f_n = @(x, lam) proj_l1_ball(x,r);


params.x0 = prox_f_n(x0);

% vec=ones(dimN,1);
% prox_f_n= @(x,lam) proj_hyperplane_box(x,vec,100,-1,1);
% params.x0=prox_f_n(params.x0);


%oracle = Oracle(f_s, f_n, grad_f_s, prox_f_n);

F.fs=f_s;
F.fn=f_n;
F.grad=grad_f_s;
F.prox=prox_f_n;

end

function result = log_exp(a, x)
    % Returns (log(1+exp(-a^T x)))
    z = -a' * x;
    result = max(z, 0) + log(exp(-abs(z)) + 1);
end

function result = log_func(a,x)
    % Returns (1/(1+exp(a^tx))
    z= a' * x;
    result = 0.5 * (1 + tanh(-0.5 * z));
end