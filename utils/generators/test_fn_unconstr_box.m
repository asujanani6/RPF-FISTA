% SPDX-License-Identifier: MIT
% Copyright © 2021 Weiwei "William" Kong

function [F, params] = test_fn_unconstr_box(N, M, m, seed, dimM, dimN, r)
% Generator of a test suite of linearly constrained nonconvex QP functions.
%
% Arguments:
%  
%   N (int): One of the objective function's hyperparameters.
%
%   dimM (int): One of the objective function's hyperparameters.
%
%   dimN (int): One of the objective function's hyperparameters.
% 
%   M (double): The target upper curvature of the objective function.
% 
%   m (double): The target lower curvature of the objective function.
% 
%   seed (int): The number used to seed MATLAB's random number generator. 
% 
% Returns:
%
%   A pair consisting of an Oracle and a struct. The oracle is first-order oracle underyling the optimization problem and the 
%   struct contains the relevant hyperparameters of the problem. 
% 

  % Initialize.
  rng(seed);
  D = sparse(1:dimN, 1:dimN, randi([1, N], 1, dimN), dimN, dimN);
  C = rand(dimM, dimN);
  B = rand(dimN, dimN);
  A = rand(dimM, dimN);
  d = rand([dimM, 1]);  
   
  % Choose (xi, tau).
  [tau, xi, ~, ~] = eigen_bisection(M, m, C, D * B);
  
  % Compute the norm of A and other factors.
  Hn = B' * (D' * D) * B;
  Hp = A' * A;
  Hn = (Hn + Hn') / 2;
  Hp = (Hp + Hp') / 2;
  norm_A = sqrt(eigs(Hp, 1, 'la')); % same as lamMax(A'*A)
   
  % Set the topology (Euclidean)
  prod_fn = @(a,b) sum(dot(a, b));
  norm_fn = @(a) norm(a, 'fro');
  
  % Compute the b vector;
  b = A * (2 * r * rand(dimN, 1) - r);
  
  % Constraint map methods.
  params.constr_fn = @(z) A * z - b;
  params.grad_constr_fn = @(z) A';
  params.set_projector = @(y) zeros(size(y));
  params.dual_cone_projector = @(y) y;
  params.K_constr = norm_A;
  
  % Basic output params.
  params.prod_fn = prod_fn;
  params.norm_fn = norm_fn;
  params.M = eigs(-xi * Hn + tau * Hp, 1, 'la');
  params.m = -eigs(-xi * Hn + tau * Hp, 1, 'sa');
  params.x0 = 2 * r * rand(dimN, 1) - r;
  
  % Special params for individual constraints.
  params.K_constr_vec = full(sqrt(sum(A .^ 2, 2)));
  params.L_constr_vec = zeros(dimM, 1);
  params.m_constr_vec = zeros(dimM, 1);
  
  % Oracle construction
  f_s = @(x) xi / 2 * norm_fn(D * B * x) ^ 2 + tau / 2 * norm_fn(C * x - d) ^ 2;
  f_n = @(x) 0;
  grad_f_s = @(x) xi * B' * (D' * D) * B * x + tau * C' * (C * x -  d);
  Hessian= @(x) xi * B' * (D' * D) * B + tau * (C' * C);
  prox_f_n = @(x, lam) box_proj(x, -r, r);
  %oracle = Oracle(f_s, f_n, grad_f_s, prox_f_n);

  F.fs=f_s;
  F.fn=f_n;
  F.grad=grad_f_s;
  F.prox=prox_f_n;
  F.hess=Hessian;
 
end