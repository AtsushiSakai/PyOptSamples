#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
Interface function to solve a quadratic programing problem with ECOS.

Author: Atsushi Sakai

"""


import numpy as np
import math
import ecos
import scipy.linalg
import scipy.sparse as sp

VERBOSE = False
#  VERBOSE = True


def ecosqp(H, f, A=None, B=None, Aeq=None, Beq=None):
    """
    solve a quadratic programing problem with ECOS

        min 1/2*x'*H*x + f'*x
        s.t. A*x <= b
             Aeq*x = beq

    return sol
        It is same data format of CVXOPT.

    """
    # ===dimension and argument checking===
    # H
    assert H.shape[0] == H.shape[1], "Hessian must be a square matrix"

    n = H.shape[0]

    # f
    if (f is None) or (f.size == 0):
        f = np.zeros((n, 1))
    else:
        assert f.shape[0] == n, "Linear term f must be a column vector of length"
        assert f.shape[1] == 1, "Linear term f must be a column vector"

    # check cholesky
    try:
        W = np.linalg.cholesky(H).T
    except np.linalg.linalg.LinAlgError:
        W = scipy.linalg.sqrtm(H)
    #  print(W)

    # set up SOCP problem
    c = np.vstack((np.zeros((n, 1)), 1.0))
    #  print(c)

    # pad Aeq with a zero column for t
    if Aeq is not None:
        Aeq = np.hstack((Aeq, np.zeros((Aeq.shape[0], 1))))
        beq = Beq
    else:
        Aeq = np.matrix([])
        beq = np.matrix([])

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
        G = Gquad
        h = hquad
        dims = {'q': [W.shape[1] + 2], 'l': 0}
    else:
        G1 = np.hstack((A, np.zeros((A.shape[0], 1))))
        G = np.vstack((G1, Gquad))
        h = np.vstack((B, hquad))
        dims = {'q': [W.shape[1] + 2], 'l': A.shape[0]}

    c = np.array(c).flatten()
    G = sp.csc_matrix(G)
    h = np.array(h).flatten()

    if Aeq.size == 0:
        sol = ecos.solve(c, G, h, dims, verbose=VERBOSE)
    else:
        Aeq = sp.csc_matrix(Aeq)
        beq = np.array(beq).flatten()
        sol = ecos.solve(c, G, h, dims, Aeq, beq, verbose=VERBOSE)
    #  print(sol)
    #  print(sol["x"])

    sol["fullx"] = sol["x"]
    sol["x"] = sol["fullx"][:n]
    sol["fval"] = sol["fullx"][-1]

    return sol


def test1():
    import cvxopt
    from cvxopt import matrix

    P = matrix(np.diag([1.0, 0.0]))
    q = matrix(np.array([3.0, 4.0]).T)
    G = matrix(np.array([[-1.0, 0.0], [0, -1.0], [-1.0, -3.0], [2.0, 5.0], [3.0, 4.0]]))
    h = matrix(np.array([0.0, 0.0, -15.0, 100.0, 80.0]).T)

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
        assert (sol["x"][i] - sol2["x"][i]) <= 0.001, "Error1"


def test2():
    import cvxopt
    from cvxopt import matrix

    P = np.matrix([[1., 0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 1., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 1., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 1., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 1., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 1., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 1., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 1., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0., 1.]])
    #  print(P.shape)

    q = np.matrix([[0.],
                   [0.],
                   [0.],
                   [0.],
                   [0.],
                   [0.],
                   [0.],
                   [0.],
                   [0.]])
    #  print(q.shape)

    A = np.matrix([[1., 0., 0., 1., 0., 0., 0., 0., 0.],
                   [-2., -0., -0., 0., 1., 0., 0., 0., 0.],
                   [0., 1., 0., -0.8, -1., 1., 0., 0., 0.],
                   [-0., -2., -0., 0., -0.9, 0., 1., 0., 0.],
                   [0., 0., 1., 0., 0., -0.8, -1., 1., 0.],
                   [-0., -0., -2., 0., 0., 0., -0.9, 0., 1.]])
    #  print(A.shape)

    B = np.matrix([[2.8],
                   [1.8],
                   [0.],
                   [0.],
                   [0.],
                   [0.]])
    #  print(B.shape)

    sol = cvxopt.solvers.qp(matrix(P), matrix(q), A=matrix(A), b=matrix(B))

    #  #  print(sol)
    print(sol["x"])
    #  #  print(sol["primal objective"])

    sol2 = ecosqp(P, q, Aeq=A, Beq=B)
    print(sol2["x"])

    for i in range(len(sol["x"])):
        print(sol["x"][i], sol2["x"][i])
        assert (sol["x"][i] - sol2["x"][i]) <= 0.001, "Error1"


def test3():
    import cvxopt
    from cvxopt import matrix

    P = np.matrix([[1., 0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 1., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 1., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 1., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 1., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 1., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 1., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 1., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0., 1.]])
    #  print(P.shape)

    q = np.matrix([[0.],
                   [0.],
                   [0.],
                   [0.],
                   [0.],
                   [0.],
                   [0.],
                   [0.],
                   [0.]])
    #  print(q.shape)

    A = np.matrix([[1., 0., 0., 1., 0., 0., 0., 0., 0.],
                   [-2., -0., -0., 0., 1., 0., 0., 0., 0.],
                   [0., 1., 0., -0.8, -1., 1., 0., 0., 0.],
                   [-0., -2., -0., 0., -0.9, 0., 1., 0., 0.],
                   [0., 0., 1., 0., 0., -0.8, -1., 1., 0.],
                   [-0., -0., -2., 0., 0., 0., -0.9, 0., 1.]])
    #  print(A.shape)

    B = np.matrix([[2.8],
                   [1.8],
                   [0.],
                   [0.],
                   [0.],
                   [0.]])
    #  print(B.shape)

    G = np.matrix([[1., 0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 1., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 1., 0., 0., 0., 0., 0., 0.],
                   [-1., -0., -0., 0., 0., 0., 0., 0., 0.],
                   [-0., -1., -0., 0., 0., 0., 0., 0., 0.],
                   [-0., -0., -1., 0., 0., 0., 0., 0., 0.]])
    print(G)

    h = np.matrix([[0.7],
                   [0.7],
                   [0.7],
                   [0.7],
                   [0.7],
                   [0.7]])

    print(h)

    sol = cvxopt.solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h), A=matrix(A), b=matrix(B))

    #  #  print(sol)
    print(sol["x"])
    #  #  print(sol["primal objective"])

    sol2 = ecosqp(P, q, A=G, B=h, Aeq=A, Beq=B)
    print(sol2["x"])

    for i in range(len(sol["x"])):
        print(sol["x"][i], sol2["x"][i])
        assert (sol["x"][i] - sol2["x"][i]) <= 0.001, "Error1"


def test4():
    import cvxopt
    from cvxopt import matrix

    P = np.matrix([[191.99932803, 171.39304576, 147.178958, 120.30997616, 92.49500206, 65.31828796, 40.35170482, 19.26589923, 3.94625875, -3.38015554],
                   [171.39304576, 167.09565179, 145.65216985, 120.99032909, 94.22693001, 67.28312768, 42.01670423, 20.35056876, 4.40430536, -3.36426],
                   [147.178958, 145.65216985, 140.48573714, 118.57442124, 94.0149707, 968.14309597, 43.16207273, 21.29176064, 4.9181394, -3.2487312],
                   [120.30997616, 120.99032909, 118.57442124, 112.92525975, 91.10905601, 67.4446326, 43.55391584, 22.0075338, 5.4951828, -2.998032],
                   [92.49500206, 94.22693001, 94.01497079, 91.10905601, 85.54225615, 64.60108256, 42.8883114, 22.3903428, 6.143988, - 2.56656],
                   [65.31828796, 67.28312768, 68.14309597, 67.4446326, 64.60108256, 59.857233, 40.7725188, 22.300068, 6.87444, -1.896],
                   [40.35170482, 42.01670423, 43.16207273, 43.55391584, 42.8883114, 40.7725188, 37.701348, 21.55524, 7.698, -0.912],
                   [19.26589923, 20.35056876, 21.29176064, 22.0075338, 22.3903428, 22.300068, 21.55524, 20.922, 8.628, 0.48],
                   [3.94625875, 4.40430536, 4.9181394, 5.4951828, 6.143988, 6.87444, 7.698, 8.628, 10.68, 2.4],
                   [-3.38015554, -3.36426, -3.2487312, -2.998032, -2.56656, -1.896, -0.912, 0.48, 2.4, 6.]])

    #  print(P.shape)

    q = np.matrix([[212.94555574],
                   [185.13212169],
                   [155.14312658],
                   [124.37387985],
                   [94.08581829],
                   [65.50068784],
                   [39.89203254],
                   [18.67890289],
                   [3.5268026],
                   [-3.53874558]])

    #  print(q.shape)

    sol = cvxopt.solvers.qp(matrix(P), matrix(q))

    #  #  print(sol)
    print(sol["x"])
    #  #  print(sol["primal objective"])

    sol2 = ecosqp(P, q)
    print(sol2["x"])

    for i in range(len(sol["x"])):
        print(sol["x"][i], sol2["x"][i])
        assert abs(sol["x"][i] - sol2["x"][i]) <= 0.001, "Error1"


def test5():
    import cvxopt
    from cvxopt import matrix

    P = np.matrix([[191.99932803, 171.39304576, 147.178958, ],
                   [171.39304576, 167.09565179, 145.65216985],
                   [147.178958, 145.65216985, 140.48573714, ]])

    #  print(P.shape)

    q = np.matrix([[212.94555574],
                   [185.13212169],
                   [-3.53874558]])

    #  print(q.shape)

    sol = cvxopt.solvers.qp(matrix(P), matrix(q))

    #  #  print(sol)
    #  #  print(sol["primal objective"])

    sol2 = ecosqp(P, q)

    print("cvxopt")
    print(sol["x"])

    print("ecosqp")
    print(sol2["x"])

    for i in range(len(sol["x"])):
        print(sol["x"][i], sol2["x"][i])
        assert abs(sol["x"][i] - sol2["x"][i]) <= 0.001, "Error1"


if __name__ == '__main__':
    #  test1()
    #  test2()
    test3()
