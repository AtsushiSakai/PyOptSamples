#! /usr/bin/python 
# -*- coding: utf-8 -*- 
u""" 
Simple sample code to solve linear programming with cvxopt

author Atsushi Sakai
"""

import numpy
import cvxopt
from cvxopt import matrix

c=matrix(numpy.array([-29.0,-45.0,0.0,0.0]))
G=matrix(numpy.diag([-1.0,-1.0,-1.0,-1.0]))
h=matrix(numpy.array([0.0,0.0,0.0,0.0]))
A=matrix(numpy.array([[2.0,8.0,1.0,0.0],[4.0,4.0,0.0,1.0]]))
b=matrix(numpy.array([60.0,60.0]))

print("c:")
print(c)
print("G:")
print(G)
print("h:")
print(h)
print("A:")
print(A)
print("b:")
print(b)

sol=cvxopt.solvers.lp(c,G,h,A,b)

print(sol["x"])
print(sol["primal objective"])

