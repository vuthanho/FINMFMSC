import numpy as np
import time


def emwnenmf_m(data, G, F, r, Tmax):
    tol = 1e-3
    delta_measure = 1
    em_iter_max = round(Tmax / delta_measure) + 1  #
    T = np.empty(shape=(em_iter_max + 1))
    T.fill(np.nan)
    RMSE = np.empty(shape=(2, em_iter_max + 1))
    RMSE.fill(np.nan)

    # RRE = np.empty(shape=(em_iter_max + 1))
    # RRE.fill(np.nan)
    M_loop = 3  # Number of passage over M step
    ITER_MAX = 50  # maximum inner iteration number (Default)
    ITER_MIN = 5  # minimum inner iteration number (Default)

    X = data.X + np.multiply(data.nW, np.dot(G, F))

    XFt = np.dot(X, F.T)
    FFt = np.dot(F, F.T)

    GtX = np.dot(G.T, X)
    GtG = np.dot(G.T, G)

    GradG = np.dot(G, FFt) - XFt
    GradF = np.dot(GtG, F) - GtX

    init_delta = stop_rule(np.hstack((G.T, F)), np.hstack((GradG.T, GradF)))
    tolF = tol * init_delta
    tolG = tolF  # Stopping tolerance

    # Iterative updating
    k = 0
    RMSE[:, k] = np.linalg.norm(F[:, 0:-1] - data.F[:, 0:-1], 2, axis=1) / np.sqrt(F.shape[1] - 1)
    T[k] = 0
    t = time.time()
    # Main loop
    while time.time() - t <= Tmax + delta_measure:

        # Estimation step
        X = data.X + np.multiply(data.nW, np.dot(G, F))

        # Maximisation step
        for _ in range(M_loop):
            FFt = F.dot(F.T)
            XFt = X.dot(F.T) - data.Phi_G.dot(FFt)
            np.put(G, data.idxOG, 0)
            G, iterG = maj_G(G, FFt, XFt, ITER_MAX, tolG, data.idxOG)
            np.put(G, data.idxOG, data.sparsePhi_G)
            if iterG <= ITER_MIN:
                tolG = 0.1 * tolG

            GtG = G.T.dot(G)
            GtX = G.T.dot(X) - GtG.dot(data.Phi_F)
            np.put(F, data.idxOF, 0)
            F, iterF = maj_F(F, GtG, GtX, ITER_MAX, tolF, data.idxOF)
            np.put(F, data.idxOF, data.sparsePhi_F)
            if iterF <= ITER_MIN:
                tolF = 0.1 * tolF

            if time.time() - t - k * delta_measure >= delta_measure:
                k = k + 1
                if k >= em_iter_max + 1:
                    break
                RMSE[:, k] = np.linalg.norm(F[:, 0:-1] - data.F[:, 0:-1], 2, axis=1) / np.sqrt(F.shape[1] - 1)
                T[k] = time.time() - t
    return {'RMSE': RMSE, 'T': T}


def stop_rule(X, GradX):
    # Stopping Criterions
    pGrad = GradX[np.any(np.dstack((X > 0, GradX < 0)), 2)]
    return np.linalg.norm(pGrad)


def maj_G(G1, FFt, XFt, ITER_MAX, tolG, idxOG):
    Y = G1
    alpha1 = 1
    L = np.linalg.norm(FFt)
    Grad_y = Y.dot(FFt) - XFt

    for i in range(1, ITER_MAX + 1):
        G2 = np.maximum(Y - (1 / L) * Grad_y, 0)
        np.put(G2, idxOG, 0)
        alpha2 = (1 + np.sqrt(4 * alpha1 ** 2 + 1)) / 2

        Y = G2 + ((alpha1 - 1) / alpha2) * (G2 - G1)
        Grad_y = Y.dot(FFt) - XFt

        G1 = G2
        alpha1 = alpha2

        if stop_rule(Y, Grad_y) <= tolG:
            break

    return G1, i


def maj_F(F1, GtG, GtX, ITER_MAX, tolF, idxOF):
    Y = F1
    alpha1 = 1
    L = np.linalg.norm(GtG)
    Grad_y = GtG.dot(Y) - GtX

    for i in range(1, ITER_MAX + 1):
        F2 = np.maximum(Y - (1 / L) * Grad_y, 0)
        np.put(F2, idxOF, 0)
        alpha2 = (1 + np.sqrt(4 * alpha1 ** 2 + 1)) / 2

        Y = F2 + ((alpha1 - 1) / alpha2) * (F2 - F1)
        Grad_y = GtG.dot(Y) - GtX

        F1 = F2
        alpha1 = alpha2

        if stop_rule(Y, Grad_y) <= tolF:
            break

    return F1, i
