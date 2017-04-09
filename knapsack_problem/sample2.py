#
# A sample code to solve a knapsack_problem with cvxpy
#
# Author: Atsushi Sakai (@Atsushi_twi)
#

import cvxpy
import numpy as np
import random
import time

N_item = 10000

size = np.array([random.randint(1, 50) for i in range(N_item)])
weight = np.array([random.randint(1, 50) for i in range(N_item)])
capacity = 100

x = cvxpy.Bool(size.shape[0])
objective = cvxpy.Maximize(weight * x)
constraints = [capacity >= size * x]

start = time.time()
prob = cvxpy.Problem(objective, constraints)
prob.solve(solver=cvxpy.ECOS_BB)
elapsed_time = time.time() - start
print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
result = [round(ix[0, 0]) for ix in x.value]

print("status:", prob.status)
print("optimal value", prob.value)
print("result x:", result)
