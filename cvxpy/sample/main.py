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
# コスト最適化
objective = Minimize(sum_squares(A*x - b))
# 制約
constraints = [0 <= x, x <= 1]
prob = Problem(objective, constraints)

result = prob.solve()
# 最適値
print "optimal parameter:\n",x.value

# ラグランジュパラメータ
print "Lagrange parameter\n",constraints[0].dual_value

#最適化の結果
print ("status:"+prob.status)


