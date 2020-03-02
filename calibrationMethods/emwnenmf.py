import numpy as np
import time

def emwnenmf(data,Ginit,Finit,r,Tmax):
    MinIter = 10
    tol     = 1e-5
    T       = np.zeros(shape=(1,301))
    RRE     = np.zeros(shape=(1,301))

    ITER_MAX=500  # maximum inner iteration number (Default)
    ITER_MIN=10   # minimum inner iteration number (Default)

    Xcomp = data.X + np.multiply(data.nW,np.dot(Ginit,Finit))

    # Compress left and right
    L,R = RSI_compression(Xcomp,r)
    X_L = np.dot(L,Xcomp)
    X_R = np.dot(Xcomp,R)

    F_comp = np.dot(Finit,R)
    G_comp = np.dot(L,Ginit)

    FXt = np.dot(F_comp,X_R.T)
    FFt = np.dot(F_comp,F_comp.T)

    GtX = np.dot(G_comp.T,X_L)
    GtG = np.dot(G_comp.T,G_comp)

    GradG = np.dot(Ginit,FFt)-FXt.T
    GradF = np.dot(GtG,Finit)-GtX

    init_delta = stop_rule(np.hstack((Ginit.T,Finit)),np.hstack((GradG.T,GradF)))
    tolF = max(tol,1e-3)*init_delta
    tolG = tolF # Stopping tolerance

    # Iterative updating
    G = Ginit.T
    k = 0
    RRE[k] = nmf_norm_fro(Xcomp,G.T,H)
    T[k] = 0
    t = time.time()
    # Main loop
    while(time.time()-t <= Tmax+0.5):
        # Estimation step
        # Maximisation step
    return {'G' : G, 'F' : F, 'RRE' : RRE, 'T': T}

def stop_rule(X,GradX):
    # Stopping Criterions
    pGrad = GradX[GradX<0|X>0]
    return np.linalg.norm(pGrad,2)

def NNLS(Z,GtG,GtX,iterMin,iterMax,tol):
    L = np.linalg.norm(GtG,2) # Lipschitz constant
    H = Z # Initialization
    Grad = np.dot(GtG,Z)-GtX #Gradient
    alpha1 = 1

    for iter in range(1,iterMax+1):
        H0 = H
        H = np.maximum(Z-Grad/L,0) # Calculate squence 'Y'
        alpha2 = 0.5*(1+np.sqrt(1+4*alpha1**2))
        Z = H + ((alpha1-1)/alpha2)*(H-H0)
        alpha1 = alpha2
        Grad=np.dot(GtG,Z)-GtX

        # Stopping criteria
        if iter>=iterMin:
            # Lin's stopping criteria
            pgn = stop_rule(Z,Grad)
            if pgn <= tol:
                break
    return H,iter,Grad


def nmf_norm_fro(X,G,F):
    f = np.square(np.linalg.norm(X-np.dot(G,F),'fro'))/np.square(np.linalg.norm(X,'fro'))
    return f

def RSI_compression(X,r):
    compressionLevel = 20
    m,n = X.shape
    l = min(n,max(compressionLevel,r+10))

    OmegaL = np.random.randn(n,l)
    Y = np.dot(X,OmegaL)
    for _ in range(4):
        Y = np.linalg.qr(Y,mode='reduced')[0]
        S = np.dot(X.T,Y)
        Z = np.linalg.qr(S,mode='reduced')[0]
        Y = np.dot(X,Z)
    L = np.linalg.qr(Y,mode='reduced')[0].T

    OmegaR = np.random.randn(l,m)
    Y = np.dot(OmegaR,X)
    for _ in range(4):
        Y = np.linalg.qr(Y.T,mode='reduced')[0]
        S = np.dot(X,Y)
        Z = np.linalg.qr(S,mode='reduced')[0]
        Y = np.dot(Z.T,X)
    R = np.linalg.qr(Y.T,mode='reduced')[0]
    return L,R