#! /usr/bin/python
# -*- coding: utf-8 -*-

#  import ecos

import numpy
import cvxopt
from cvxopt import matrix


def ecosqp(H, f, A, b, Aeq=None, beq=None, LB=None, UB=None):
    pass


def test1():
    P = matrix(numpy.diag([1.0, 0.0]))
    q = matrix(numpy.array([3.0, 4.0]))
    G = matrix(numpy.array([[-1.0, 0.0], [0, -1.0], [-1.0, -3.0], [2.0, 5.0], [3.0, 4.0]]))
    h = matrix(numpy.array([0.0, 0.0, -15.0, 100.0, 80.0]))

    sol = cvxopt.solvers.qp(P, q, G, h)

    #  print(sol)
    print(sol["x"])
    #  print(sol["primal objective"])

    assert sol["x"][0] - 0.0, "Error1"
    assert sol["x"][1] - 5.0, "Error1"


if __name__ == '__main__':
    test1()
