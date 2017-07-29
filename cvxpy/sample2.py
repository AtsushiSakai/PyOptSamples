import cvxpy

z1 = cvxpy.Variable()
z2 = cvxpy.Variable()

objective = cvxpy.Minimize(cvxpy.abs(z1 + 5) + cvxpy.abs(z2 - 3))

constraints = [2.5 <= z1, z1 <= 5]
constraints += [-1.0 <= z2, z2 <= 1]
prob = cvxpy.Problem(objective, constraints)

result = prob.solve()

# 最適値
print("z1", z1.value)
print("z2", z2.value)
print("cost:", prob.value)
