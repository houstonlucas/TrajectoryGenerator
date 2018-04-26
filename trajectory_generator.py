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


class TrajectoryGenerator:
    def __init__(self, keyframes, n, ndims, k_r):
        self.n = n
        self.keyframes = keyframes
        self.m = len(keyframes) - 1
        self.ndims = ndims
        self.dim_size = self.n * self.m
        self.k_r = k_r

        self.Q = self.create_Q_matrix(k_r)

    def generate_trajectory(self):
        A_list_pos, b_list_pos = self.create_position_constraints()
        A_list_deriv_coupling, b_list_deriv_coupling = self.create_deriv_coupling_constraints()

        init_A = []
        init_b = []
        A_list_deriv_initial, b_list_deriv_initial = self.create_deriv_constraint(0, 10.0, t=0.0, dim_index=1, k=1)
        init_A.append(A_list_deriv_initial)
        init_b.append(b_list_deriv_initial)
        A_list_deriv_initial, b_list_deriv_initial = self.create_deriv_constraint(0, 0.0, t=0.0, dim_index=0, k=1)
        init_A.append(A_list_deriv_initial)
        init_b.append(b_list_deriv_initial)

        A_list = A_list_pos + A_list_deriv_coupling + init_A
        b_list = b_list_pos + b_list_deriv_coupling + init_b

        c_size = self.ndims * self.dim_size
        A = matrix(np.vstack(A_list))
        b = matrix(np.vstack(b_list))

        p = matrix(np.zeros(c_size))
        G = matrix(np.zeros((1, c_size)))
        h = matrix(np.zeros(1))

        sol = solvers.qp(self.Q, p, G, h, A, b, kktsolver='ldl', options={'kktreg': 1e-9})
        return np.array(sol['x'])

    # n- number of terms in polynomials
    # m- number of segments in trajectory
    # j- segment index for this constraint
    # constraint- the value that is the constraint
    # t- the time at which the constraint must be met
    # dim_index- determines which dimension of the keyframe to constrain
    # ndim- the number of dimensions of the parametric equations
    # is_begining_constraint- Bool determining if this is a constraint on the beginning or end of the segment
    def create_position_constraint(self, j, constraint_value, t, dim_index, is_beginning_constraint):
        # Calculate how many leading and trailing zeros are in the matrix row
        num_leading_dims = dim_index
        num_trailing_dims = self.ndims - (dim_index + 1)
        num_leading_polys = j
        num_trailing_polys = self.m - (j + 1)

        if is_beginning_constraint:
            num_leading_polys -= 1
            num_trailing_polys += 1
        num_leading_zeros = num_leading_dims * self.dim_size + num_leading_polys * self.n
        num_trailing_zeros = num_trailing_dims * self.dim_size + num_trailing_polys * self.n

        leading = [0.0] * num_leading_zeros
        trailing = [0.0] * num_trailing_zeros
        constraint_coefficients = [t ** i for i in range(self.n)]
        A = np.matrix(leading + constraint_coefficients + trailing)
        b = np.matrix([constraint_value])
        return A, b

    def create_deriv_constraint(self, j, constraint_value, t, dim_index, k):
        num_leading_dims = dim_index
        num_trailing_dims = self.ndims - (dim_index + 1)
        num_leading_polys = j
        num_trailing_polys = self.m - (j + 1)

        num_leading_zeros = num_leading_dims * self.dim_size + num_leading_polys * self.n
        num_trailing_zeros = num_trailing_dims * self.dim_size + num_trailing_polys * self.n

        leading = [0.0] * num_leading_zeros
        trailing = [0.0] * num_trailing_zeros

        poly = [1.0] * self.n
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

    def create_deriv_coupling_constraint(self, j, t, dim_index, k):
        num_leading_dims = dim_index
        num_trailing_dims = self.ndims - (dim_index + 1)
        num_leading_polys = j - 1
        num_trailing_polys = self.m - (j + 1)

        num_leading_zeros = num_leading_dims * self.dim_size + num_leading_polys * self.n
        num_trailing_zeros = num_trailing_dims * self.dim_size + num_trailing_polys * self.n

        leading = [0.0] * num_leading_zeros
        trailing = [0.0] * num_trailing_zeros

        poly = [1.0] * self.n
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

    def create_position_constraints(self):
        A_list = []
        b_list = []
        # Setup Positional constraints
        for dim in range(self.ndims):
            for j in range(self.m):
                time = self.keyframes[j][-1]
                if j > 0:
                    pos = self.keyframes[j][:-1]
                    A_temp, b_temp = self.create_position_constraint(j, pos[dim], time, dim, True)
                    A_list.append(A_temp)
                    b_list.append(b_temp)
                if j < self.m:
                    pos = self.keyframes[j][:-1]
                    A_temp, b_temp = self.create_position_constraint(j, pos[dim], time, dim, False)
                    A_list.append(A_temp)
                    b_list.append(b_temp)
        return A_list, b_list

    # Creates the constraints on the derivatives between polynomials
    def create_deriv_coupling_constraints(self):
        A_list = []
        b_list = []
        k = 1
        for dim in range(self.ndims):
            # Do not use the end points
            for j in range(1, self.m):
                time = self.keyframes[j][-1]

                A_temp, b_temp = self.create_deriv_coupling_constraint(j, time, dim, k)
                A_list.append(A_temp)
                b_list.append(b_temp)

        return A_list, b_list

    def parse_polys_from_vector(self, poly):
        polys = []
        for dim in range(self.ndims):
            for j in range(self.m):
                components = []
                for i in range(self.n):
                    index = dim * self.dim_size + j * self.n + i
                    components.append(round(poly[index], 4))
                polys.append(components)
        return polys

    def pretty_print_poly(self, poly):
        for dim in range(self.ndims):
            for j in range(self.m):
                components = []
                for i in range(self.n):
                    index = dim * self.dim_size + j * self.n + i
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

    def pretty_print_matrix(self, A):
        A = np.array(A)
        for row in A:
            s = ""
            for val in row:
                s += str(val) + " \t "
            print(s)

    def create_Q_matrix(self, k_r):
        # Used for indexing outer product matrix
        d = self.n - k_r

        poly = [1.0] * self.n
        d2_poly = deriv_k(poly, k_r)

        d2_poly_squared = np.outer(d2_poly, d2_poly)

        e1 = apply_integration(d2_poly_squared, 0.0)
        e2 = apply_integration(d2_poly_squared, 1.0)
        diff = (e2 - e1)[:d, :d]

        fill = [0.0] * k_r
        Q_subpart = spdiag(fill + [matrix(diff)])
        double_diagonal(Q_subpart)
        Q_part = [Q_subpart] * self.m
        # TODO: this doesn't currently work with k_yaw
        Q = spdiag(Q_part * self.ndims)
        return Q


def main():
    # Order of derivative on r
    k_r = 4
    # Order of polynomials
    n = 5

    ndims = 2

    # Keyframes stored as (x,y,z,psi,t) triples
    keyframes = [(3.0, 4.0, -1.0, -1.0, 0.0), (-2.0, 5.0, -2.0, 0.0, 1.0),
                 (0.0, -1.0, 0.0, 1.5, 2.0)]  # , (2.0, -2.0, 3.0)]

    tgen = TrajectoryGenerator(keyframes, n, ndims, k_r)

    x = tgen.generate_trajectory()

    for poly in tgen.parse_polys_from_vector(x):
        print(poly)


if __name__ == '__main__':
    main()
