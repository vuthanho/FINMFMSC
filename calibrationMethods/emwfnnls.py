import numpy as np
import time
from numpy import float64, sum,  nonzero, newaxis, finfo


nu = newaxis


def emwfnnls(data, G, F, r, Tmax):
    tol = 0
    delta_measure = 1
    em_iter_max = round(Tmax / delta_measure) + 1  #
    T = np.empty(shape=(em_iter_max + 1))
    T.fill(np.nan)
    RMSE = np.empty(shape=(2, em_iter_max + 1))
    RMSE.fill(np.nan)

    # RRE = np.empty(shape=(em_iter_max + 1))
    # RRE.fill(np.nan)

    np.put(F, data.idxOF, data.sparsePhi_F)
    np.put(G, data.idxOG, data.sparsePhi_G)
    X = data.X + np.multiply(data.nW, np.dot(G, F))

    GtX = np.dot(G.T, X)
    GtG = np.dot(G.T, G)

    # Iterative updating
    G = G.T
    k = 0
    RMSE[:, k] = np.linalg.norm(F[:, 0:-1] - data.F[:, 0:-1], 2, axis=1) / np.sqrt(F.shape[1] - 1)
    T[k] = 0
    t = time.time()
    # Main loop
    while time.time() - t <= Tmax + delta_measure:

        # Estimation step
        X = data.X + np.multiply(data.nW, np.dot(G.T, F))

        # Maximisation step
        # Optimize F with fixed G
        np.put(F, data.idxOF, 0)
        F, _ = fnnls(GtG, GtX - GtG.dot(data.Phi_F))
        np.put(F, data.idxOF, data.sparsePhi_F)
        FFt = np.dot(F, F.T)
        FXt = np.dot(F, X.T)

        # Optimize G with fixed F
        np.put(G.T, data.idxOG, 0)
        G, _ = fnnls(FFt, FXt - FFt.dot(data.Phi_G.T))
        np.put(G.T, data.idxOG, data.sparsePhi_G)
        GtG = np.dot(G, G.T)
        GtX = np.dot(G, X)

        if time.time() - t - k * delta_measure >= delta_measure:
            k = k + 1
            if k >= em_iter_max + 1:
                break
            RMSE[:, k] = np.linalg.norm(F[:, 0:-1] - data.F[:, 0:-1], 2, axis=1) / np.sqrt(F.shape[1] - 1)
            T[k] = time.time() - t
            # RRE[k] = nmf_norm_fro(data.Xtheo, G.T, F, data.W)
            # if k%100==0:
            #     print(str(k)+'   '+str(RMSE[0,k])+'   '+str(RMSE[1,k]))
    # plt.semilogy(RRE)
    # plt.show()
    return {'RMSE': RMSE, 'T': T}


def stop_rule(X, GradX):
    # Stopping Criterions
    pGrad = GradX[np.any(np.dstack((X > 0, GradX < 0)), 2)]
    return np.linalg.norm(pGrad, 2)


# machine epsilon
eps = finfo(float64).eps


def any(a):
    # assuming a vector, a
    larger_than_zero = sum(a > 0)
    if larger_than_zero:
        return True
    else:
        return False


def find_nonzero(a):
    # returns indices of nonzero elements in a
    return nonzero(a)[0]


def fnnls(AtA, Aty, epsilon=None, iter_max=None):
    """
    Given a matrix A and vector y, find x which minimizes the objective function
    f(x) = ||Ax - y||^2.
    This algorithm is similar to the widespread Lawson-Hanson method, but
    implements the optimizations described in the paper
    "A Fast Non-Negativity-Constrained Least Squares Algorithm" by
    Rasmus Bro and Sumen De Jong.
    Note that the inputs are not A and y, but are
    A^T * A and A^T * y
    This is to avoid incurring the overhead of computing these products
    many times in cases where we need to call this routine many times.
    :param AtA:       A^T * A. See above for definitions. If A is an (m x n)
                      matrix, this should be an (n x n) matrix.
    :type AtA:        numpy.ndarray
    :param Aty:       A^T * y. See above for definitions. If A is an (m x n)
                      matrix and y is an m dimensional vector, this should be an
                      n dimensional vector.
    :type Aty:        numpy.ndarray
    :param epsilon:   Anything less than this value is consider 0 in the code.
                      Use this to prevent issues with floating point precision.
                      Defaults to the machine precision for doubles.
    :type epsilon:    float
    :param iter_max:  Maximum number of inner loop iterations. Defaults to
                      30 * [number of cols in A] (the same value that is used
                      in the publication this algorithm comes from).
    :type iter_max:   int, optional
    """
    if epsilon is None:
        epsilon = np.finfo(np.float64).eps

    n = AtA.shape[0]

    if iter_max is None:
        iter_max = 30 * n

    if Aty.ndim != 1 or Aty.shape[0] != n:
        raise ValueError('Invalid dimension; got Aty vector of size {}, ' \
                         'expected {}'.format(Aty.shape, n))

    # Represents passive and active sets.
    # If sets[j] is 0, then index j is in the active set (R in literature).
    # Else, it is in the passive set (P).
    sets = np.zeros(n, dtype=np.bool)
    # The set of all possible indices. Construct P, R by using `sets` as a mask
    ind = np.arange(n, dtype=int)
    P = ind[sets]
    R = ind[~sets]

    x = np.zeros(n, dtype=np.float64)
    w = Aty
    s = np.zeros(n, dtype=np.float64)

    i = 0
    # While R not empty and max_(n \in R) w_n > epsilon
    while not np.all(sets) and np.max(w[R]) > epsilon and i < iter_max:
        # Find index of maximum element of w which is in active set.
        j = np.argmax(w[R])
        # We have the index in MASKED w.
        # The real index is stored in the j-th position of R.
        m = R[j]

        # Move index from active set to passive set.
        sets[m] = True
        P = ind[sets]
        R = ind[~sets]

        # Get the rows, cols in AtA corresponding to P
        AtA_in_p = AtA[P][:, P]
        # Do the same for Aty
        Aty_in_p = Aty[P]

        # Update s. Solve (AtA)^p * s^p = (Aty)^p
        s[P] = np.linalg.lstsq(AtA_in_p, Aty_in_p, rcond=None)[0]
        s[R] = 0.

        while np.any(s[P] <= epsilon):
            i += 1

            mask = (s[P] <= epsilon)
            alpha = np.min(x[P][mask] / (x[P][mask] - s[P][mask]))
            x += alpha * (s - x)

            # Move all indices j in P such that x[j] = 0 to R
            # First get all indices where x == 0 in the MASKED x
            zero_mask = (x[P] < epsilon)
            # These correspond to indices in P
            zeros = P[zero_mask]
            # Finally, update the passive/active sets.
            sets[zeros] = False
            P = ind[sets]
            R = ind[~sets]

            # Get the rows, cols in AtA corresponding to P
            AtA_in_p = AtA[P][:, P]
            # Do the same for Aty
            Aty_in_p = Aty[P]

            # Update s. Solve (AtA)^p * s^p = (Aty)^p
            s[P] = np.linalg.lstsq(AtA_in_p, Aty_in_p, rcond=None)[0]
            s[R] = 0.

        x = s.copy()
        w = Aty - AtA.dot(x)

    return x
