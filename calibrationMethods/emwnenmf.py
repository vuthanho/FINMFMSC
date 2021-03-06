import numpy as np
import time


def emwnenmf(data, G, F, r, Tmax):
    tol = 1e-3
    delta_measure = 1
    em_iter_max = round(Tmax / delta_measure) + 1  #
    T = np.empty(shape=(em_iter_max + 1))
    T.fill(np.nan)
    RMSE = np.empty(shape=(2, em_iter_max + 1))
    RMSE.fill(np.nan)

    # RRE = np.empty(shape=(em_iter_max + 1))
    # RRE.fill(np.nan)
    M_loop = 2  # Number of passage over M step
    ITER_MAX = 100  # maximum inner iteration number (Default)
    ITER_MIN = 5  # minimum inner iteration number (Default)

    X = data.X + np.multiply(data.nW, np.dot(G, F))

    FXt = np.dot(F, X.T)
    FFt = np.dot(F, F.T)

    GtX = np.dot(G.T, X)
    GtG = np.dot(G.T, G)

    GradG = np.dot(G, FFt) - FXt.T
    GradF = np.dot(GtG, F) - GtX

    init_delta = stop_rule(np.hstack((G.T, F)), np.hstack((GradG.T, GradF)))
    tolF = tol * init_delta
    tolG = tolF  # Stopping tolerance

    # Iterative updating
    G = G.T
    GtX = np.dot(G, X) - GtG.dot(data.Phi_F)
    k = 0
    RMSE[:, k] = np.linalg.norm(F[:, 0:-1] - data.F[:, 0:-1], 2, axis=1) / np.sqrt(F.shape[1] - 1)
    T[k] = 0
    t = time.time()
    # Main loop
    while time.time() - t <= Tmax + delta_measure:

        # Estimation step
        if (k+1) % M_loop:
            X = data.X + np.multiply(data.nW, np.dot(G.T, F))

        # Maximisation step
        # Optimize F with fixed G
        np.put(F, data.idxOF, 0)
        F, iterF, _ = NNLS(F, GtG, GtX, ITER_MAX, tolF, data.idxOF, False)
        np.put(F, data.idxOF, data.sparsePhi_F)
        # print(F[:,0:5])
        if iterF <= ITER_MIN:
            tolF = tolF / 10
            # print('Tweaked F tolerance to '+str(tolF))
        FFt = np.dot(F, F.T)
        FXt = np.dot(F, X.T) - FFt.dot(data.Phi_G.T)

        # Optimize G with fixed F
        np.put(G.T, data.idxOG, 0)
        G, iterG, _ = NNLS(G, FFt, FXt, ITER_MAX, tolG, data.idxOG, True)
        np.put(G.T, data.idxOG, data.sparsePhi_G)
        if iterG <= ITER_MIN:
            tolG = tolG / 10
            # print('Tweaked G tolerance to '+str(tolG))
        GtG = np.dot(G, G.T)
        GtX = np.dot(G, X) - GtG.dot(data.Phi_F)

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
    return np.linalg.norm(pGrad, 2)


def NNLS(Z, GtG, GtX, iterMax, tol, idxfixed, transposed):
    L = np.linalg.norm(GtG, 2)  # Lipschitz constant
    H0 = Z  # Initialization
    Grad = np.dot(GtG, Z) - GtX  # Gradient
    alpha1 = 1

    for iter in range(1, iterMax + 1):
        H = np.maximum(Z - (1 / L) * Grad, 0)  # Calculate squence 'Y'

        if transposed:  # If Z = G.T
            np.put(H.T, idxfixed, 0)
        else:  # If Z = F
            np.put(H, idxfixed, 0)
        alpha2 = 0.5 * (1 + np.sqrt(1 + 4 * alpha1 ** 2))
        Z = H + ((alpha1 - 1) / alpha2) * (H - H0)
        alpha1 = alpha2
        Grad = np.dot(GtG, Z) - GtX
        H0 = H
        # Lin's stopping criteria
        pgn = stop_rule(Z, Grad)
        if pgn <= tol:
            break
    return H, iter, Grad


def nmf_norm_fro(X, G, F, *args):
    W = args
    if len(W) == 0:
        f = np.square(np.linalg.norm(X - np.dot(G, F), 'fro')) / np.square(np.linalg.norm(X, 'fro'))
    else:
        W = W[0]
        f = np.square(np.linalg.norm(X - np.multiply(W, np.dot(G, F)), 'fro')) / np.square(np.linalg.norm(X, 'fro'))
    return f
