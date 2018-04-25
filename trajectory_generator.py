import copy
from cvxopt import spdiag, matrix, solvers
import numpy as np


# Polynomials are stored from lowest ordered coefficients to highest.
# e.g. 5t^2 + 3t + 8 is represented as [8, 3, 5]

def deriv_k(poly, k):
    if k > 1:
        return deriv(deriv_k(poly, k - 1))
    else:
        return deriv(poly)


def deriv(poly):
    l = len(poly)
    return [0.0 if i + 1 == l else (i + 1) * poly[i + 1] for i in range(l)]


def apply_integration(M, t):
    h, w = M.shape
    M = copy.copy(M)
    for i in range(h):
        for j in range(w):
            c = float(i + j + 1)
            M[i, j] = M[i, j] * (t ** c) / c
    return M


def double_diagonal(M):
    l = M.size[0]
    for i in range(l):
        M[i, i] *= 2.0


# n- number of terms in polynomials
# m- number of segments in trajectory
# j- segment index for this constraint
# constraint- the value that is the constraint
# t- the time at which the constraint must be met
# dim_index- determines which dimension of the keyframe to constrain
# ndim- the number of dimensions of the parametric equations
# is_begining_constraint- Bool determining if this is a constraint on the beginning or end of the segment
def create_position_constraint(n, m, j, constraint_value, t, dim_index, ndim, is_beginning_constraint):
    # Calculate how many leading and trailing zeros are in the matrix row
    dim_size = n * m
    num_leading_dims = dim_index
    num_trailing_dims = ndim - (dim_index + 1)
    num_leading_polys = j
    num_trailing_polys = m - (j + 1)

    if is_beginning_constraint:
        num_leading_polys -= 1
        num_trailing_polys += 1
    num_leading_zeros = num_leading_dims * dim_size + num_leading_polys * n
    num_trailing_zeros = num_trailing_dims * dim_size + num_trailing_polys * n

    leading = [0.0] * num_leading_zeros
    trailing = [0.0] * num_trailing_zeros
    constraint_coefficients = [t ** i for i in range(n)]
    A = np.matrix(leading + constraint_coefficients + trailing)
    b = np.matrix([constraint_value])
    return A, b


def create_deriv_constraint(n, m, j, constraint_value, t, dim_index, ndim, k):
    dim_size = n * m
    num_leading_dims = dim_index
    num_trailing_dims = ndim - (dim_index + 1)
    num_leading_polys = j
    num_trailing_polys = m - (j + 1)

    num_leading_zeros = num_leading_dims * dim_size + num_leading_polys * n
    num_trailing_zeros = num_trailing_dims * dim_size + num_trailing_polys * n

    leading = [0.0] * num_leading_zeros
    trailing = [0.0] * num_trailing_zeros

    poly = [1.0] * n
    kth_deriv = deriv_k(poly, k)
    kth_deriv = [value * (t ** i) for i, value in enumerate(kth_deriv)]

    # Adjustments below because I defined the derivative function in a way that doesn't work well here.
    # The adjustment made below re-aligns the coefficients to the correct locations
    # TODO: fix this (note this deriv change would affect how the Q matrix is generated.
    kth_deriv = kth_deriv[-k:] + kth_deriv[:-k]

    constraint_coefficients = kth_deriv
    A = np.matrix(leading + constraint_coefficients + trailing)
    b = np.matrix([constraint_value])

    return A, b


def create_deriv_coupling_constraint(n, m, j, t, dim_index, ndim, k):
    dim_size = n * m
    num_leading_dims = dim_index
    num_trailing_dims = ndim - (dim_index + 1)
    num_leading_polys = j - 1
    num_trailing_polys = m - (j + 1)

    num_leading_zeros = num_leading_dims * dim_size + num_leading_polys * n
    num_trailing_zeros = num_trailing_dims * dim_size + num_trailing_polys * n

    leading = [0.0] * num_leading_zeros
    trailing = [0.0] * num_trailing_zeros

    poly = [1.0] * n
    kth_deriv = deriv_k(poly, k)
    kth_deriv = [value * (t ** i) for i, value in enumerate(kth_deriv)]

    # Adjustments below because I defined the derivative function in a way that doesn't work well here.
    # The adjustment made below re-aligns the coefficients to the correct locations
    # TODO: fix this (note this deriv change would affect how the Q matrix is generated.
    kth_deriv = kth_deriv[-k:] + kth_deriv[:-k]
    neg_kth_deriv = [-1 * value for value in kth_deriv]

    constraint_coefficients = kth_deriv + neg_kth_deriv
    A = np.matrix(leading + constraint_coefficients + trailing)
    b = np.matrix([0.0])

    return A, b


def create_position_constraints(n, m, ndims, keyframes):
    A_list = []
    b_list = []
    # Setup Positional constraints
    for dim in range(ndims):
        for j in range(m):
            time = keyframes[j][-1]
            if j > 0:
                pos = keyframes[j][:-1]
                A_temp, b_temp = create_position_constraint(n, m - 1, j, pos[dim], time, dim, ndims, True)
                A_list.append(A_temp)
                b_list.append(b_temp)
            if j < m - 1:
                pos = keyframes[j][:-1]
                A_temp, b_temp = create_position_constraint(n, m - 1, j, pos[dim], time, dim, ndims, False)
                A_list.append(A_temp)
                b_list.append(b_temp)
    return A_list, b_list


# Creates the constraints on the derivatives between polynomials
def create_deriv_coupling_constraints(n, m, ndims, keyframes):
    A_list = []
    b_list = []
    k = 1
    for dim in range(ndims):
        # Do not use the end points
        for j in range(1, m - 1):
            time = keyframes[j][-1]

            A_temp, b_temp = create_deriv_coupling_constraint(n, m - 1, j, time, dim, ndims, k)
            A_list.append(A_temp)
            b_list.append(b_temp)

    return A_list, b_list


def parse_polys_from_vector(poly, n, m, ndims):
    dim_size = n * m
    polys = []
    for dim in range(ndims):
        for j in range(m):
            components = []
            for i in range(n):
                index = dim * dim_size + j * n + i
                components.append(round(poly[index], 4))
            polys.append(components)
    return polys


def pretty_print_poly(poly, n, m, ndims):
    dim_size = n * m
    for dim in range(ndims):
        for j in range(m):
            components = []
            for i in range(n):
                index = dim * dim_size + j * n + i
                coefficient = poly[index]
                if 'e' in str(coefficient):
                    parts = str(poly[index]).split("e")
                    parts[0] = str(round(float(parts[0]), 3))
                    coefficient = parts[0] + "*10^({})".format(parts[1])
                if i == 0:
                    components.append(str(coefficient))
                else:
                    component = "{}t^{}".format(coefficient, i)
                    components.append(component)
            print(" + ".join(components))


def pretty_print_matrix(A):
    A = np.array(A)
    for row in A:
        s = ""
        for val in row:
            s += str(val) + " \t "
        print(s)


def create_Q_matrix(n, m, k_r, ndims):
    # Used for indexing outer product matrix
    d = n - k_r

    poly = [1.0] * n
    d2_poly = deriv_k(poly, k_r)

    d2_poly_squared = np.outer(d2_poly, d2_poly)

    e1 = apply_integration(d2_poly_squared, 0.0)
    e2 = apply_integration(d2_poly_squared, 1.0)
    diff = (e2 - e1)[:d, :d]

    fill = [0.0] * k_r
    Q_subpart = spdiag(fill + [matrix(diff)])
    double_diagonal(Q_subpart)
    Q_part = [Q_subpart] * (m - 1)
    # TODO: this doesn't currently work with k_yaw
    Q = spdiag(Q_part * ndims)
    return Q


def main():
    # Order of derivative on r
    k_r = 4
    # Order of polynomials
    n = 5

    ndims = 2

    # Keyframes stored as (x,y,z,psi,t) triples
    keyframes = [(3.0, 4.0, -1.0, -1.0,  0.0), (-2.0, 5.0, -2.0, 0.0, 1.0), (0.0, -1.0, 0.0, 1.5, 2.0)]  # , (2.0, -2.0, 3.0)]
    m = len(keyframes)

    Q = create_Q_matrix(n, m, k_r, ndims)
    print(Q.size)

    A_list_pos, b_list_pos = create_position_constraints(n, m, ndims, keyframes)
    A_list_deriv_coupling, b_list_deriv_coupling = create_deriv_coupling_constraints(n, m, ndims, keyframes)

    init_A = []
    init_b = []
    A_list_deriv_initial, b_list_deriv_initial = create_deriv_constraint(n, m - 1, 0, 10.0,
                                                                         t=0.0, dim_index=1, ndim=ndims, k=1)
    init_A.append(A_list_deriv_initial)
    init_b.append(b_list_deriv_initial)
    A_list_deriv_initial, b_list_deriv_initial = create_deriv_constraint(n, m - 1, 0, 0.0,
                                                                         t=0.0, dim_index=0, ndim=ndims, k=1)
    init_A.append(A_list_deriv_initial)
    init_b.append(b_list_deriv_initial)

    A_list = A_list_pos + A_list_deriv_coupling + init_A
    b_list = b_list_pos + b_list_deriv_coupling + init_b

    print("######")
    c_size = ndims * n * (m - 1)
    A = matrix(np.vstack(A_list))
    b = matrix(np.vstack(b_list))

    p = matrix(np.zeros(c_size))
    G = matrix(np.zeros((1, c_size)))
    h = matrix(np.zeros(1))

    sol = solvers.qp(Q, p, G, h, A, b, kktsolver='ldl', options={'kktreg': 1e-9})

    x = np.array(sol['x'])

    for poly in parse_polys_from_vector(x, n, m - 1, ndims):
        print(poly)


if __name__ == '__main__':
    main()
