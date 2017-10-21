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
    cij = [2, 5, 3, 1, 2, 0, 2, 4]
    inf = 100000.0
    uij = [inf, inf, inf, inf, 1, inf, inf, inf]
    bi = [-5, 10, 0, -2, -3]

    print("ni", ni)
    print("nj", nj)
    print("cij", cij)
    print("uij", uij)
    print("bi", bi)

    x = cvxpy.Variable(len(ni))
    objective = cvxpy.Minimize(np.transpose(cij) * x)
    constraints = [0 <= x, x <= uij]
    for iin in range(1, len(bi) + 1):
        inp = sum([x[i - 1] for i in ni if i == iin])
        out = sum([x[i - 1] for i in nj if i == iin])
        constraints += [(inp - out) == bi[iin - 1]]

    prob = cvxpy.Problem(objective, constraints)

    result = prob.solve()
    print("Opt result:", result)
    print("optimal parameter:\n", x.value)
    print("Lagrange parameter\n", constraints[0].dual_value)
    print("status:" + prob.status)


if __name__ == '__main__':
    main()
