import numpy as np
import time
import matplotlib.pyplot as plt


def emwmunmf(data, G, F, r, Tmax):
    delta_measure = 1
    em_iter_max = round(Tmax / delta_measure) + 1  #
    T = np.empty(shape=(em_iter_max + 1))
    T.fill(np.nan)
    RMSE = np.empty(shape=(2, em_iter_max + 1))
    RMSE.fill(np.nan)

    # RRE = np.empty(shape=(em_iter_max + 1))
    # RRE.fill(np.nan)
    secu = 1e-12
    M_loop = 5
    np.put(F, data.idxOF, data.sparsePhi_F)
    np.put(G, data.idxOG, data.sparsePhi_G)

    delta_G = G
    delta_F = F

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
            # Optimize F with fixed G
            np.put(delta_G, data.idxOG, 0)
            delta_G = np.divide(
                np.multiply(
                    delta_G,
                    np.dot(
                        secu_plus(X - data.Phi_G.dot(F), secu),
                        F.T
                    )
                ),
                np.dot(
                    delta_G.dot(F),
                    F.T
                )
            )
            delta_G[np.isnan(delta_G)] = 0
            G = delta_G
            np.put(G, data.idxOG, data.sparsePhi_G)

            # Optimize G with fixed F
            np.put(F, data.idxOF, 0)
            delta_F = np.divide(
                np.multiply(
                    delta_F,
                    np.dot(
                        G.T,
                        secu_plus(X - G.dot(data.Phi_F), secu)
                    )
                ),
                np.dot(
                    G.T,
                    G.dot(delta_F)
                )
            )
            delta_F[np.isnan(delta_F)] = 0
            F = delta_F
            np.put(F, data.idxOF, data.sparsePhi_F)

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


def secu_plus(tutu, s):
    toto = np.maximum(tutu, s)
    toto[np.isnan(tutu)] = 0
    return toto


def nmf_norm_fro(X, G, F, *args):
    W = args
    if len(W) == 0:
        f = np.square(np.linalg.norm(X - np.dot(G, F), 'fro')) / np.square(np.linalg.norm(X, 'fro'))
    else:
        W = W[0]
        f = np.square(np.linalg.norm(X - np.multiply(W, np.dot(G, F)), 'fro')) / np.square(np.linalg.norm(X, 'fro'))
    return f
