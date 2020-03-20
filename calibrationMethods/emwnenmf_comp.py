import numpy as np
import time


def emwnenmf_comp(data, G, F, r, Tmax):
    OutIter = 50
    tol = 1e-5
    delta_measure = 1
    em_iter_max = round(Tmax / delta_measure) + 1  #
    T = np.empty(shape=(em_iter_max + 1))
    T.fill(np.nan)
    RMSE = np.empty(shape=(2, em_iter_max + 1))
    RMSE.fill(np.nan)

    ITER_MAX = 200  # maximum inner iteration number (Default)
    ITER_MIN = 10  # minimum inner iteration number (Default)

    np.put(F, data.idxOF, data.sparsePhi_F)
    np.put(G, data.idxOG, data.sparsePhi_G)
    Xcomp = data.X + np.multiply(data.nW, np.dot(G, F))

    FXt = np.dot(F, Xcomp.T)
    FFt = np.dot(F, F.T)

    GtX = np.dot(G.T, Xcomp)
    GtG = np.dot(G.T, G)

    GradG = np.dot(G, FFt) - FXt.T
    GradF = np.dot(GtG, F) - GtX

    init_delta = stop_rule(np.hstack((G.T, F)), np.hstack((GradG.T, GradF)))
    tolF = tol * init_delta
    tolG = tolF  # Stopping tolerance

    # Iterative updating
    G = G.T
    k = 0
    RMSE[:, k] = np.linalg.norm(F[:, 0:-1] - data.F[:, 0:-1], 2, axis=1) / np.sqrt(F.shape[1] - 1)
    T[k] = 0
    t = time.time()
    to_comp = False
    trigger_comp = 1e-5
    # Main loop
    while time.time() - t <= Tmax + delta_measure:

        # Estimation step
        Xcomp = data.X + np.multiply(data.nW, np.dot(G.T, F))
        if to_comp:
            # Compress left and right
            L, R = RSI_compression(Xcomp, r)
            X_L = np.dot(L, Xcomp)
            X_R = np.dot(Xcomp, R)
        # Maximisation step
        for _ in range(OutIter):
            # Optimize F with fixed G
            np.put(F, data.idxOF, 0)
            F, iterF, _ = NNLS(F, GtG, GtX - GtG.dot(data.Phi_F), ITER_MIN, ITER_MAX, tolF, data.idxOF, False)
            np.put(F, data.idxOF, data.sparsePhi_F)
            if iterF <= ITER_MIN:
                tolF = tolF / 10
                # print('Tweaked F tolerance to '+str(tolF))
            if to_comp:
                F_comp = F.dot(R)
                FFt = np.dot(F_comp, F_comp.T)
                FXt = np.dot(F_comp, X_R.T)
            else:
                FFt = np.dot(F, F.T)
                FXt = np.dot(F, Xcomp.T)

            # Optimize G with fixed F
            np.put(G.T, data.idxOG, 0)
            G, iterG, _ = NNLS(G, FFt, FXt - FFt.dot(data.Phi_G.T), ITER_MIN, ITER_MAX, tolG, data.idxOG, True)
            np.put(G.T, data.idxOG, data.sparsePhi_G)
            if iterG <= ITER_MIN:
                tolG = tolG / 10
                # print('Tweaked G tolerance to '+str(tolG))
            if to_comp:
                G_comp = G.dot(L.T)
                GtG = np.dot(G_comp, G_comp.T)
                GtX = np.dot(G_comp, X_L)
            else:
                GtG = np.dot(G, G.T)
                GtX = np.dot(G, Xcomp)

            if time.time() - t - k * delta_measure >= delta_measure:
                k = k + 1
                if k >= em_iter_max + 1:
                    break
                RMSE[:, k] = np.linalg.norm(F[:, 0:-1] - data.F[:, 0:-1], 2, axis=1) / np.sqrt(F.shape[1] - 1)
                T[k] = time.time() - t
                # if k%100==0:
                #     print(str(k)+'   '+str(RMSE[0,k])+'   '+str(RMSE[1,k]))
            if not to_comp:
                if nmf_norm_fro(Xcomp, G.T, F) > trigger_comp:
                    break
                else:
                    to_comp = True
                    print(time.time()-t)
                    break

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


def RSI_compression(X, r):
    compressionLevel = 20
    m, n = X.shape
    l = min(n, max(compressionLevel, r + 10))

    OmegaL = np.random.randn(n, l)
    Y = np.dot(X, OmegaL)
    for _ in range(4):
        Y = np.linalg.qr(Y, mode='reduced')[0]
        S = np.dot(X.T, Y)
        Z = np.linalg.qr(S, mode='reduced')[0]
        Y = np.dot(X, Z)
    L = np.linalg.qr(Y, mode='reduced')[0].T

    OmegaR = np.random.randn(l, m)
    Y = np.dot(OmegaR, X)
    for _ in range(4):
        Y = np.linalg.qr(Y.T, mode='reduced')[0]
        S = np.dot(X, Y)
        Z = np.linalg.qr(S, mode='reduced')[0]
        Y = np.dot(Z.T, X)
    R = np.linalg.qr(Y.T, mode='reduced')[0]
    return L, R
