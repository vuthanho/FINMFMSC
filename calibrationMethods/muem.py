import numpy as np
import time

def muem(data, G, F, r, Tmax):
    delta_measure = 1
    iter_max = round(Tmax / delta_measure) + 1
    T = np.empty(shape=(iter_max + 1))
    T.fill(np.nan)
    RMSE = np.empty(shape=(2, iter_max + 1))
    RMSE.fill(np.nan)

    mu_rate = 0.15
    mu_res = mu(data, G, F, r, mu_rate * Tmax, T, RMSE, delta_measure)
    return em(data, mu_res['G'], mu_res['F'], r, (1 - mu_rate) * Tmax, mu_res['T'], mu_res['RMSE'], mu_res['mu_state'], delta_measure)


def mu(data, G, F, r, Tmax, T, RMSE, delta_measure):
    W2 = np.square(data.W)
    secu = 1e-12
    # RRE = np.empty(shape=(iter_max + 1))
    # RRE.fill(np.nan)
    delta_G = G
    delta_F = F
    t = time.time()
    T[0] = time.time() - t
    RMSE[:, 0] = np.linalg.norm(F[:, 0:-1] - data.F[:, 0:-1], 2, axis=1) / np.sqrt(F.shape[1] - 1)
    i = 0
    while time.time() - t <= Tmax + delta_measure:

        # Updating G
        np.put(delta_G, data.idxOG, 0)
        delta_G = np.divide(
            np.multiply(
                delta_G,
                np.dot(
                    np.multiply(
                        W2,
                        secu_plus(data.X - data.Phi_G.dot(F), secu)
                    ),
                    F.T
                )
            ),
            np.dot(
                np.multiply(
                    W2,
                    delta_G.dot(F)
                ),
                F.T
            )
        )
        delta_G[np.isnan(delta_G)] = 0
        G = delta_G
        np.put(G, data.idxOG, data.sparsePhi_G)

        # Updating F
        np.put(F, data.idxOF, 0)
        delta_F = np.divide(
            np.multiply(
                delta_F,
                np.dot(
                    G.T,
                    np.multiply(
                        W2,
                        secu_plus(data.X - G.dot(data.Phi_F), secu)
                    )
                )
            ),
            np.dot(
                G.T,
                np.multiply(
                    W2,
                    G.dot(delta_F)
                )
            )
        )
        delta_F[np.isnan(delta_F)] = 0
        F = delta_F
        np.put(F, data.idxOF, data.sparsePhi_F)

        # Saving results for this iteration
        if time.time() - t - i * delta_measure >= delta_measure:
            i = i + 1
            RMSE[:, i] = np.linalg.norm(F[:, 0:-1] - data.F[:, 0:-1], 2, axis=1) / np.sqrt(F.shape[1] - 1)
            T[i] = time.time() - t
    return {'G': G, 'F': F, 'T': T, 'RMSE': RMSE, 'mu_state': i}


def secu_plus(tutu, s):
    toto = np.maximum(tutu, s)
    toto[np.isnan(tutu)] = 0
    return toto


def em(data, G, F, r, Tmax, T, RMSE, mu_state, delta_measure):

    tol = 1e-5
    em_iter_max = round(Tmax / delta_measure) + 1  #

    ITER_MAX = 3  # maximum inner iteration number (Default)
    ITER_MIN = 1  # minimum inner iteration number (Default)

    np.put(F, data.idxOF, data.sparsePhi_F)
    np.put(G, data.idxOG, data.sparsePhi_G)
    X = data.X + np.multiply(data.nW, np.dot(G, F))

    FXt = np.dot(F, X.T)
    FFt = np.dot(F, F.T)

    GtX = np.dot(G.T, X)
    GtG = np.dot(G.T, G)

    tolF = tol * stop_rule(F, np.dot(GtG, F) - GtX)
    tolG = tol * stop_rule(G.T, (np.dot(G, FFt) - FXt.T).T)  # Stopping tolerance

    # Iterative updating
    G = G.T
    k = mu_state
    t = time.time()

    # Main loop
    while time.time() - t <= Tmax + delta_measure:

        # Estimation step
        X = data.X + np.multiply(data.nW, np.dot(G.T, F))

        # Maximisation step
        # Optimize F with fixed G
        np.put(F, data.idxOF, 0)
        F, iterF, _ = NNLS(F, GtG, GtX - GtG.dot(data.Phi_F), ITER_MIN, ITER_MAX, tolF, data.idxOF, False)
        np.put(F, data.idxOF, data.sparsePhi_F)
        # print(F[:,0:5])
        if iterF <= ITER_MIN:
            tolF = tolF / 10
            # print('Tweaked F tolerance to '+str(tolF))
        FFt = np.dot(F, F.T)
        FXt = np.dot(F, X.T)

        # Optimize G with fixed F
        np.put(G.T, data.idxOG, 0)
        G, iterG, _ = NNLS(G, FFt, FXt - FFt.dot(data.Phi_G.T), ITER_MIN, ITER_MAX, tolG, data.idxOG, True)
        np.put(G.T, data.idxOG, data.sparsePhi_G)
        if iterG <= ITER_MIN:
            tolG = tolG / 10
            # print('Tweaked G tolerance to '+str(tolG))
        GtG = np.dot(G, G.T)
        GtX = np.dot(G, X)

        if time.time() - t - (k-mu_state) * delta_measure >= delta_measure:
            k = k + 1
            if (k-mu_state) >= em_iter_max:
                break
            RMSE[:, k] = np.linalg.norm(F[:, 0:-1] - data.F[:, 0:-1], 2, axis=1) / np.sqrt(F.shape[1] - 1)
            T[k] = T[mu_state] + time.time() - t
    return {'RMSE': RMSE, 'T': T}


def stop_rule(X, GradX):
    # Stopping Criterions
    pGrad = GradX[np.any(np.dstack((X > 0, GradX < 0)), 2)]
    return np.linalg.norm(pGrad, 2)


def NNLS(Z, GtG, GtX, iterMin, iterMax, tol, idxfixed, transposed):
    L = np.linalg.norm(GtG, 2)  # Lipschitz constant
    H = Z  # Initialization
    Grad = np.dot(GtG, Z) - GtX  # Gradient
    alpha1 = 1

    for iter in range(1, iterMax + 1):
        H0 = H
        H = np.maximum(Z - Grad / L, 0)  # Calculate squence 'Y'

        if transposed:  # If Z = G.T
            np.put(H.T, idxfixed, 0)
        else:  # If Z = F
            np.put(H, idxfixed, 0)
        alpha2 = 0.5 * (1 + np.sqrt(1 + 4 * alpha1 ** 2))
        Z = H + ((alpha1 - 1) / alpha2) * (H - H0)
        alpha1 = alpha2
        Grad = np.dot(GtG, Z) - GtX

        # Stopping criteria
        if iter >= iterMin:
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
