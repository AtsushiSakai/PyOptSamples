#! /usr/bin/python 
# -*- coding: utf-8 -*- 
u""" 
Simple sample code to solve quadratic programming with cvxopt

author Atsushi Sakai
"""

import numpy
import cvxopt
from cvxopt import matrix

P=matrix(numpy.diag([1.0,0.0]))
q=matrix(numpy.array([3.0,4.0]))
G=matrix(numpy.array([[-1.0,0.0],[0,-1.0],[-1.0,-3.0],[2.0,5.0],[3.0,4.0]]))
h=matrix(numpy.array([0.0,0.0,-15.0,100.0,80.0]))

print(P)
print(q)
print(G)
print(h)

sol=cvxopt.solvers.qp(P,q,G,h)

print(sol)
print(sol["x"])
print(sol["primal objective"])

