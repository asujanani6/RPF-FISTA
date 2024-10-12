% SPDX-License-Identifier: MIT
% Copyright © 2021 Weiwei "William" Kong

function [F, params] = test_simplexHenry(seed,m,n,density)
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
A = sprand(m,n,density);
support_density = .1;
x_true = full(sprandn(n, 1, support_density));
b = A*x_true;

z0=randn(n,1);
z0=z0/norm(z0);
x0=z0.*z0;

% Set the topology (Euclidean)
prod_fn = @(a,b) sum(dot(a, b));
norm_fn = @(a) norm(a, 'fro');

% Output params
params.prod_fn = prod_fn;
params.norm_fn = norm_fn;
params.M1=norm(full(A))^2;
params.m = -1;

% params.M2 = norm(A'*b)^2/4;
% params.M3=norm(A, 2)^2/4;

% Create the Oracle object
f_s = @(x) 0.5*(params.norm_fn(A*x-b))^2;
f_n = @(x) 0;
grad_f_s = @(x) transpose(A)*(A*x-b);
prox_f_n = @(x, lam) TKPsimplex_proj(x,1);


params.x0 = x0;

% vec=ones(dimN,1);
% prox_f_n= @(x,lam) proj_hyperplane_box(x,vec,100,-1,1);
% params.x0=prox_f_n(params.x0);


%oracle = Oracle(f_s, f_n, grad_f_s, prox_f_n);

F.fs=f_s;
F.fn=f_n;
F.grad=grad_f_s;
F.prox=prox_f_n;

end

%From Ting Kei Pong, Polytechnique Hong Kong
function d = TKPsimplex_proj(c,tau)

n = max(size(c));
p = -c;
pmax = max(p);
sm = sum(p);
if sm >= n*pmax - tau
    lambda = (tau+sm)/n;
    d = max(0, c + lambda);
    clear p;
    return;
end

p = sort(p);

sm = 0;
for i = 1:n-1
    smnew = sm + i*(p(i+1) - p(i));
    if smnew >= tau
        break
    end
    sm = smnew;
end

k = i;
delta = (tau - sm)/k;
lambda = p(k) + delta;
d = max(0, c + lambda);
%clear p;
end