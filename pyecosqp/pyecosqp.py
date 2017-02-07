#! /usr/bin/python
# -*- coding: utf-8 -*-

#  import ecos

import numpy as np
import cvxopt
import scipy.linalg
import math
from cvxopt import matrix
import ecos
import scipy.sparse as sp

"""
function [X,fval,exitflag,output,lambda,Tsolve,c,G,h,dims,Aeq,beq] = ecosqp(H,f,A,B,Aeq,Beq,lb,ub,opts)
% ECOSQP - solve quadratic program using the ECOS second-order cone solver.
% The interface mimics MATLAB's quadprog interface.
%
%   X = ECOSQP(H,f,A,b) attempts to solve the quadratic program
%
%            minimize 1/2*x'*H*x + f'*x
%          subject to A*x <= b
%
%   X = ECOSQP(H,f,A,b,Aeq,beq) attempts to solve the QP from above
%   with additional equality constraints on variable x:
%
%            minimize 1/2*x'*H*x + f'*x
%          subject to A*x <= b
%                     Aeq*x = beq
%
%
%   X = ECOSQP(H,f,A,b,Aeq,beq,LB,UB) attempts to solve the problem with
%   lower and upper bounds on the design variables, X, so that the solution
%   is in the range LB <= X <= UB. Use empty matrices for LB and UB if no
%   bounds exist. Set LB(i) = -Inf if X(i) is unbounded below; set
%   UB(i) = +Inf if X(i) is unbounded above.
%
%   X = ECOSQP(...,OPTIONS) solves the QP as above and uses the settings as
%   defined in the OPTIONS structure. See ECOSOPTIMSET for more information
%   on how to set settings in ECOS.
%
%   [X,FVAL] = ECOSQP(...) returns in addition to the minimizer X also
%   the optimal function value of the QP.
%
%   [X,FVAL,EXITFLAG] = ECOSQP(...) returns the exitflag with the following
%   meaning:
%
%      1  KKT optimality conditions satisfied - optimal solution found.
%      0  Maximum number of iterations exceeded.
%     -2  Problem is (primal) infeasible.
%     -3  Problem is unbounded (dual infeasible).
%
%
%   [X,FVAL,EXITFLAG,INFO] = ECOSQP(...) returns in addition the
%   INFO struct as returned for the converted problem by ECOS. See the ECOS
%   documentation for more help on the particular fields of the INFO
%   struct.
%
%   [X,FVAL,EXITFLAG,INFO,TSOLVE] = ECOSQP(...) returns the runtime for
%   solving the converted problem as returned by ECOS. Note that this can
%   be different from a "tic; ecosqp(...); toc;" since an initial "square
%   root factorization" of the Hessian H is to be performed first.
%
%
% This file is an interface for ECOS, and rewrites the QP as a second-order
% cone program (SOCP) that can be solved by ECOS. The rewriting occurs
% through an epigraph reformulation of the objective function,
%
%   ||       Wx              ||
%   || (t - f'*x + 1)/sqrt(2)||_2 <= (t-f'*x-1)/sqrt(2) <==> 1/2*x'*H*x + f'*x <= t,
%
% assuming that the Hessian H is positive definite, and therefore admits a
% Cholesky decomposition H = W'*W. The following problem is solved when
% calling ECOS after the rewriting:
%
%     minimize    t
%     subject to  A*x <= b
%           [ lb <= x <= ub ]
%           [ Aeq*x == beq    ]
%
%           ||       Wx     ||
%           || (t-f'*x+1)/sqrt(2) ||_2 <= (t-f'*x-1)/sqrt(2)
%
%
% If the Hessian H is positive semi-definite, an eigenvalue decomposition
% is computed to obtain W=H^(1/2). This might be slow for large matrices,
% but work well for diagonal Hessians. Eigenvalues which are zero are
% discarded from the computation of W.
%
%
% (c) A. Domahidi, ETH Zurich & embotech GmbH, Zurich, Switzerland, 2012-15.



if( nargin < 8 )
    ub = [];
end

if( nargin < 7 )
    lb = [];
end

if( nargin == 7 && isstruct(lb) )
    opts = lb;
    lb = [];
end

if( nargin == 8 && isstruct(ub) )
    opts = ub;
    ub = [];
end

if( nargin < 5 )
    Aeq = [];
    Beq = [];
end

if( nargin == 5 )
    if( ~isstruct(Aeq) )
        error('Either Aeq or an options struct expected as fifth argument');
    else
        opts = Aeq;
        Aeq = [];
    end
end

if( nargin < 3 )
    A = [];
    B = [];
end

if( ~exist('opts','var') )
    opts.verbose = 1;
end

%% display that we can begin with conversion
if( opts.verbose > 0 )
    disp('ECOSQP: Converting QP to SOCP...');
end

%% precondition
scale = max(abs(f));
scale = 1;
f = f./scale;
H = H./scale;


%% check if Cholesky decomposition of H exists,
%  i.e. whether we have a positive definite Hessian
assert( ~isempty(H),'Quadratic programming requires a Hessian.');
try
    tic
    W = chol(H,'upper');
    if( opts.verbose > 0 )
        fprintf('ECOSQP: Time for Cholesky: %4.2f seconds\n', toc);
    end
catch %#ok<CTCH>
    warning('Hessian not positive definite, using sqrt(H) instead of chol');
    W = sqrt(H);
    k = 0;
    eliminateIdx = [];
    for i = 1:size(W,1)
        if( all(W(i,:) == 0) )
            k = k+1;
            eliminateIdx(k) = i; %#ok<AGROW>
        end
    end
    W(eliminateIdx,:) = [];
    if( opts.verbose > 0 ), fprintf('%d zero rows in square root of Hessian eliminated\n',k-1); end
end


%% set up SOCP problem
% The new variable is stacked as [x, t]
c = [zeros(n,1); 1];

% upper bounds
if( exist('ub','var') && ~isempty(ub) )
    % find indices which are upper bounded
    Aub = speye(n);
    Aub(isinf(ub),:) = [];
    Bub = ub( ~isinf(ub) );
    A = [A; Aub];
    B = [B; Bub];
end

% lower bounds
if( exist('lb','var') && ~isempty(lb) )
    % find indices which are lower bounded
    Alb = -speye(n);
    Alb(isinf(lb),:) = [];
    Blb = -lb( ~isinf(lb) );
    A = [A; Alb];
    B = [B; Blb];
end

% pad Aeq with a zero column for t
if( ~isempty(Aeq) )
    Aeq = [Aeq, zeros(size(Aeq,1),1)];
    beq = Beq;
else
    Aeq = [];
    beq = [];
end


% create second-order cone constraint for objective function
fhalf = f./sqrt(2);
zerocolumn = zeros(size(W,1),1);
Gquad = [fhalf', -1/sqrt(2);
         -W, zerocolumn;
         -fhalf', +1/sqrt(2)];
hquad = [1/sqrt(2); zerocolumn; 1/sqrt(2)];
if( isempty(A) )
    G = Gquad;
    h = hquad;
    dims.l = 0;
    dims.q = size(W,1)+2;
else
    G = [A, zeros(size(A,1),1); Gquad];
    h = [B; hquad];
    dims.l = size(A,1);
    dims.q = size(W,1)+2;
end

%% sparsify
G = sparse(G);
Aeq = sparse(Aeq);


%% solve
if( opts.verbose > 0 ), fprintf('Conversion completed. Calling ECOS...\n'); end
if( isempty(Aeq) )
    save noscale c G h dims opts
    [x,y,info,s,z] = ecos(c,G,h,dims,opts); %#ok<ASGLU>
else
    [x,y,info,s,z] = ecos(c,G,h,dims,Aeq,beq,opts); %#ok<ASGLU>
end


%% prepare return variables
X = x(1:n)*scale;
fval = x(end);
switch( info.exitflag )
    case 1, exitflag = -2;
    case 2, exitflag = -3;
    case 0, exitflag = 1;
    otherwise, exitflag = -100;
end
output.statusstring = info.infostring;
output.iterations = info.iter;
output.time = info.timing.runtime;
lambda = [z(1:dims.l); y];
Tsolve = info.timing.runtime;
output.ecosinfo = info;
"""


def ecosqp(H, f, A, B, Aeq=None, Beq=None, LB=None, UB=None):
    ret = {}

    # ===dimension and argument checking===
    # H
    if (f is None) or (H.size == 0):
        print("Error:Quadratic programming requires a non-empty, non-zero Hessian")
        return ret
    elif H.shape[0] != H.shape[1]:
        print("Error:Hessian must be a square matrix")
        return ret

    n = H.shape[0]
    #  print(n)

    # f
    if (f is None) or (f.size == 0):
        f = np.zeros((n, 1))
    else:
        assert f.shape[0] == n, "Linear term f must be a column vector of length"
        assert f.shape[1] == 1, "Linear term f must be a column vector"

    # check cholesky
    try:
        W = np.linalg.cholesky(H)
    except np.linalg.linalg.LinAlgError:
        print("cannot calc W")
        W = scipy.linalg.sqrtm(H)
    #  print(W)

    # set up SOCP problem
    # The new variable is stacked as [x, t]
    c = np.vstack((np.zeros((n, 1)), 1.0))
    #  print(c)

    # pad Aeq with a zero column for t
    if Aeq is not None:
        Aeq = np.vstack((Aeq, np.zeros(Aeq.shape[1], 1)))
        #  beq = Beq
    else:
        Aeq = np.matrix([])
        #  beq = np.matrix([])

    #  print(Aeq)
    #  print(beq)

    # create second-order cone constraint for objective function
    fhalf = f / math.sqrt(2.0)
    #  print(fhalf)
    zerocolumn = np.zeros((W.shape[1], 1))
    #  print(zerocolumn)

    tmp = 1.0 / math.sqrt(2.0)

    Gquad1 = np.hstack((fhalf.T, np.matrix(-tmp)))
    Gquad2 = np.hstack((-W, zerocolumn))
    Gquad3 = np.hstack((-fhalf.T, np.matrix(tmp)))
    Gquad = np.vstack((Gquad1, Gquad2, Gquad3))
    #  print(Gquad1)
    #  print(Gquad2)
    #  print(Gquad3)
    #  print(Gquad)

    hquad = np.vstack((tmp, zerocolumn, tmp))
    #  print(hquad)

    if A is None:
        print("N-------")
    else:
        G1 = np.hstack((A, np.zeros((A.shape[0], 1))))
        G = np.vstack((G1, Gquad))
        h = np.vstack((B, hquad))
        dims = {'q': [W.shape[1] + 2], 'l': A.shape[0]}

    c = np.array(c).flatten()
    G = sp.csc_matrix(G)
    h = np.array(h).flatten()

    sol = ecos.solve(c, G, h, dims)
    #  print(sol)
    #  print(sol["x"])

    sol["fullx"] = sol["x"]
    sol["x"] = sol["fullx"][:n]
    sol["fval"] = sol["fullx"][-1]

    return sol


def test1():
    P = matrix(np.diag([1.0, 0.0]))
    q = matrix(np.array([3.0, 4.0]))
    G = matrix(np.array([[-1.0, 0.0], [0, -1.0], [-1.0, -3.0], [2.0, 5.0], [3.0, 4.0]]))
    h = matrix(np.array([0.0, 0.0, -15.0, 100.0, 80.0]))

    sol = cvxopt.solvers.qp(P, q, G, h)

    #  print(sol)
    print(sol["x"])
    #  print(sol["primal objective"])

    assert sol["x"][0] - 0.0, "Error1"
    assert sol["x"][1] - 5.0, "Error2"

    P = np.diag([1.0, 0.0])
    q = np.matrix([3.0, 4.0]).T
    G = np.matrix([[-1.0, 0.0], [0, -1.0], [-1.0, -3.0], [2.0, 5.0], [3.0, 4.0]])
    h = np.matrix([0.0, 0.0, -15.0, 100.0, 80.0]).T

    sol2 = ecosqp(P, q, G, h)

    for i in range(len(sol["x"])):
        assert sol["x"][i] - sol2["x"][i], "Error1"


if __name__ == '__main__':
    test1()
