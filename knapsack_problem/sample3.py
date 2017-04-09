#
# A sample code to solve a knapsack_problem with cvxpy
#
# Author: Atsushi Sakai (@Atsushi_twi)
#

import cvxpy
import numpy as np

size = np.array([21, 11, 15, 9, 34, 25, 41, 52])
weight = np.array([22, 12, 16, 10, 35, 26, 42, 53])
capacity = 100

x = cvxpy.Int(size.shape[0])
objective = cvxpy.Maximize(weight * x)
constraints = [capacity >= size * x]
constraints += [x >= 0]

prob = cvxpy.Problem(objective, constraints)
prob.solve(solver=cvxpy.ECOS_BB)
result = [round(ix[0, 0]) for ix in x.value]

print("status:", prob.status)
print("optimal value", prob.value)
print("size :", size * x.value)
print("result x:", result)
