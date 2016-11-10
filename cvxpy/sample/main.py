#! /usr/bin/python 
# -*- coding: utf-8 -*- 

from cvxpy import *
import numpy
import matplotlib.pyplot as plt


z1 = Variable()
z2 = Variable()

objective = Minimize(abs(z1+5)+abs(z2-3))

constraints = [2.5 <= z1, z1 <= 5]
constraints += [-1.0 <= z2, z2 <= 1]
prob = Problem(objective, constraints)

result = prob.solve()

# 最適値
print "z1",z1.value
print "z2",z2.value
print "cost:",prob.value

