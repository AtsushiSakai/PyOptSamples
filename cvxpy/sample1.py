import cvxpy
import numpy

m = 10
n = 5
numpy.random.seed(1)
A = numpy.random.randn(m, n)
b = numpy.random.randn(m)

x = cvxpy.Variable(n)
objective = cvxpy.Minimize(cvxpy.sum_squares(A * x - b))
constraints = [0 <= x, x <= 1]
prob = cvxpy.Problem(objective, constraints)

result = prob.solve()
print("optimal parameter:\n", x.value)
print("Lagrange parameter\n", constraints[0].dual_value)
print("status:" + prob.status)
