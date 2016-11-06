#! /usr/bin/python 
# -*- coding: utf-8 -*- 

from cvxpy import *
import numpy
import matplotlib.pyplot as plt

m = 30
n = 20
numpy.random.seed(1)
A = numpy.random.randn(m, n)
b = numpy.random.randn(m)

x = Variable(n)
# $B%3%9%H:GE,2=(B
objective = Minimize(sum_squares(A*x - b))
# $B@)Ls(B
constraints = [0 <= x, x <= 1]
prob = Problem(objective, constraints)

result = prob.solve()
# $B:GE,CM(B
print "optimal parameter:\n",x.value

# $B%i%0%i%s%8%e%Q%i%a!<%?(B
print "Lagrange parameter\n",constraints[0].dual_value

#$B:GE,2=$N7k2L(B
print ("status:"+prob.status)


