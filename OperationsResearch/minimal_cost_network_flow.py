"""

Minimal Cost Network Flow

author: Atsushi Sakai

"""

import cvxpy
import numpy as np


def main():
    print(__file__ + " start!!")

    ni = [1, 1, 2, 3, 3, 4, 4, 5]
    nj = [2, 3, 3, 4, 5, 1, 5, 2]
    cij = np.array([4, 2, 1, 1, 2, 5, 2, 4])
    inf = 100000.0
    uij = [inf, inf, inf, inf, 1, inf, inf, inf]
    bi = [-5, 11, -1, -2, -3]

    print("ni", ni)
    print("nj", nj)
    print("cij", cij)
    print("uij", uij)
    print("bi", bi)

    x = cvxpy.Variable(len(ni))
    objective = cvxpy.Minimize(cij.T * x)
    constraints = [0 <= x, x <= uij]
    for iin in range(1, len(bi) + 1):
        inp = sum([x[i - 1] for i in range(1, len(ni)) if ni[i - 1] == iin])
        out = sum([x[i - 1] for i in range(1, len(ni)) if nj[i - 1] == iin])
        constraints += [inp - out == bi[iin - 1]]

    prob = cvxpy.Problem(objective, constraints)

    result = prob.solve(solver=cvxpy.ECOS)
    print("Opt result:", result)
    print("optimal parameter:\n", [int(i) for i in x.value])
    print("status:" + prob.status)


if __name__ == '__main__':
    main()
