% SPDX-License-Identifier: MIT
% Copyright © 2021 Weiwei "William" Kong

function [F, params] = test_fn_log_det(seed,n,l,u)
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

Z=rand(n,n);
x0=rand(n,n);
x0=transpose(x0)*x0;

Y=(transpose(Z)*Z)+2*eye(n,n);

% Set the topology (Euclidean)
prod_fn = @(a,b) sum(dot(a, b));
norm_fn = @(a) norm(a, 'fro');

% Output params
params.prod_fn = prod_fn;
params.norm_fn = norm_fn;



% params.M2 = norm(A'*b)^2/4;
% params.M3=norm(A, 2)^2/4;

% Create the Oracle object
f_s = @(X) trace(Y*X)-log(det(X));
f_n = @(X) 0;
grad_f_s = @(X) Y-inv(X);
prox_f_n = @(X, lam) proj_spectral_box_sym(X,l,u);


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

